import os
import time
import pygame
import pyaudio
import wave
import webrtcvad  # [필수] VAD 라이브러리 추가
from openai import OpenAI

class DriverChatbot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.is_processing = False 
        
        # 콜백 함수 저장소
        self.callback_result = None  # 결과 처리 (인정/부정)
        self.callback_start = None   # 대화 시작 알림 (임계값 완화용)
        self.callback_end = None     # 대화 종료 알림 (임계값 복구용)
        
        self.stop_signal = False 
        
        # [VAD를 위한 오디오 설정 변경]
        # webrtcvad는 16000Hz, 32000Hz, 48000Hz 만 지원 (16000 권장)
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  
        self.FRAME_DURATION_MS = 30  # 30ms 단위로 쪼개서 검사
        self.CHUNK = int(self.RATE * self.FRAME_DURATION_MS / 1000) # 480 프레임
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"
        self.ALARM_FILE = "sounds/alarm.wav"
        
        # [VAD 설정] 모드 3: 매우 둔감함 (잡음 많은 차 안 환경용)
        self.vad = webrtcvad.Vad(3)

        # [최종 프롬프트]
        self.system_prompt = """
        너는 졸음운전 방지 AI 조수야. 친구처럼 자연스러운 '반말'을 써.
        답변은 1~2문장으로 짧게 해.

        [상태별 응답 지침 - 반드시 태그를 붙여]
        1. 운전자가 **'처음 경고'에 대해 졸음을 부정**하거나, **대화를 거부**하면: [부정] 태그를 붙여. (시스템이 DENY로 처리하고 초기화됨)
        
        2. 운전자가 졸음을 **인정**하거나 **대화로 잠을 깨우는 게 필요**하면: [인정] 태그를 붙여. (대화가 지속됨)
        
        3. [인정] 태그를 쓴 후에는 운전자가 잠을 깨도록 퀴즈를 던지거나 주요 관심사를 물어보는 등 소통을 해봐.
        
        4. **[핵심 수정]** 운전자가 **"됐어", "그만해", "꺼줘" 등 대화의 목적 달성 후 종료를 요청**하면: [종료] 태그를 붙여. (깔끔하게 시스템 종료됨)

        [예시]
        - 유저: "나 안 잤어!" -> "그래? 알겠어. 안전 운전해! [부정]"
        - 유저: "됐어 이제 그만해" -> "알겠어! 안전 운전하고 다음에 또 얘기하자. [종료]" 
        - 유저: "아 좀 졸리네..." -> "위험한데! 그럼 내가 수수께끼 하나 내줄게. [인정]"
        - 유저: "아 좀 졸리네..." -> "졸리면 안 돼! 네가 좋아하는 영화는 뭐야? [인정]"

        5. 이것외의 상황에서는 자연스럽게 대화를 이어가. 필요하다면 답변을 길게 해도 좋아.다만 태그가 붙으면 반드시 지켜야 해.
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"❌ Audio Init Error: {e}")

    # 콜백 설정 함수들
    def set_callbacks(self, on_result, on_start, on_end):
        self.callback_result = on_result
        self.callback_start = on_start
        self.callback_end = on_end

    def stop(self):
        print("🛑 [Chatbot] 강제 정지 명령 수신!")
        self.stop_signal = True
        self.is_processing = False
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except: pass

    def play_audio(self, file_path, timeout=None):
        if not os.path.exists(file_path): return
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if self.stop_signal:
                    pygame.mixer.music.stop()
                    return
                if timeout and (time.time() - start_time > timeout):
                    pygame.mixer.music.stop()
                    return
                pygame.time.Clock().tick(10)
        except: pass

    def tts_play(self, text):
        if self.stop_signal: return
        
        # 태그 제거 후 음성 출력
        clean_text = text.replace("[대화]", "").replace("[부정]", "").replace("[인정]", "").replace("[종료]", "").strip()
        if not clean_text: return
        try:
            response = self.client.audio.speech.create(model="tts-1", voice="onyx", input=clean_text)
            response.stream_to_file(self.TTS_OUTPUT)
            
            self.messages.append({"role": "assistant", "content": clean_text})
            self.play_audio(self.TTS_OUTPUT)
            
        except Exception as e:
            print(f"TTS Error: {e}")

    # [핵심 변경] VAD 방식 녹음 (말할 때만 녹음)
    def record_audio(self):
        if self.stop_signal: return False
        
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            print("\n🎤 말씀하세요... (목소리가 들리면 녹음을 시작합니다)")
            
            frames = []
            silence_duration = 0      # 침묵 지속 시간
            has_spoken = False        # 말을 시작했는지 여부
            
            # 최대 10초 대기 (무한 루프 방지)
            start_time = time.time()
            
            while True:
                if self.stop_signal: break
                if time.time() - start_time > 10: # 10초 동안 아무 말 없으면 종료
                    break

                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # VAD로 사람 목소리인지 판별
                is_speech = self.vad.is_speech(data, self.RATE)
                
                if is_speech:
                    if not has_spoken:
                        print("⚡ 감지됨! 녹음 시작...")
                        has_spoken = True
                    frames.append(data)
                    silence_duration = 0 # 말하는 중이면 침묵 초기화
                else:
                    # 소리가 안 날 때
                    if has_spoken:
                        # 이미 말을 시작했다면 -> 문장 끝인지 간보기
                        frames.append(data) 
                        silence_duration += self.FRAME_DURATION_MS
                        
                        # 1.0초 이상 조용하면 말 끝난 것으로 간주
                        if silence_duration > 1000: 
                            print("✅ 녹음 완료 (문장 끝)")
                            break
                    else:
                        # 아직 말 안 했으면 계속 듣기만 함
                        pass

            stream.stop_stream()
            stream.close()
            
            if self.stop_signal: return False
            
            # 녹음된 내용이 너무 짧으면(0.5초 미만) 잡음으로 간주하고 버림
            if len(frames) < (self.RATE / self.CHUNK * 0.5):
                return False

            wf = wave.open(self.WAVE_OUTPUT, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Record Error: {e}")
            return False
        finally: p.terminate()

    def stt(self):
        if self.stop_signal: return None
        try:
            if not os.path.exists(self.WAVE_OUTPUT): return None

            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            text = transcript.text.strip()
            
            # [안전장치] VAD를 써도 생길 수 있는 유튜브 환각 필터링
            hallucinations = [
                "Thanks for watching", "MBC", "SBS", "News", "Subtitles", 
                "구독", "좋아요", "알림 설정", "시청해", "다음 영상", "영상에서 만나요",
                "Thank you for watching", "Please like", "Subscribe", "Notification",
            ]
            for h in hallucinations:
                if h.lower() in text.lower():
                    print(f"👻 [Ghost Filter] 환각 텍스트 감지됨(무시): {text}")
                    return None

            return text
        except: return None

    def get_gpt_response(self, user_text):
        if self.stop_signal: return ""
        try:
            self.messages.append({"role": "user", "content": user_text})
            # 대화 맥락 유지 (최근 10개)
            if len(self.messages) > 10:
                self.messages = [self.messages[0]] + self.messages[-9:]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except: return "[종료] 오류"

    def start_conversation(self):
        if self.is_processing: return
        self.is_processing = True
        self.stop_signal = False
        
        # [신규] 한 대화 세션 내에서 중복으로 점수가 올라가는 것을 방지하는 플래그
        admitted_once = False 
        # 연속 침묵 횟수 및 인식 실패 횟수 카운트
        silence_error_count = 0
        # 연속 인식 실패 횟수 카운트
        stt_error_count = 0
        # [중요] 대화 시작 알림 -> Analyzer가 임계값을 완화함 (2초 -> 4초)
        if self.callback_start: self.callback_start()

        try:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            # 대화 횟수 반복 (기본 5회)
            for _ in range(5): 
                if self.stop_signal: break

                # 연속 침묵 2회 ->  종료
                if silence_error_count >= 2:
                    print("  연속 침묵 3회로 대화 종료")
                    self.tts_play("조용하네... 다음에 또 이야기하자!")
                    break
                
                # 연속 3회 인식 실패 -> 종료
                if stt_error_count >= 2:
                    print("  연속 인식 실패 3회로 대화 종료")
                    self.tts_play("음성 인식에 문제가 있나봐. 다음에 또 이야기하자!")
                    break
                
                # [변경] VAD 녹음 시도
                if not self.record_audio(): 
                    # 말을 안 했거나 잡음만 있었으면 silent_error_count 증가 
                    silence_error_count += 1
                    continue

                # 말을 했으면 침묵 카운트 초기화
                silence_error_count = 0
                
                user_text = self.stt()
                if self.stop_signal: break

                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 1:
                    stt_error_count += 1
                    print("  음성 인식 실패. 다시 시도해주세요.")

                    if stt_error_count < 3:
                        self.tts_play("잘 못 알아들었어. 다시 말해줄래?")
                        continue
                    else:
                        continue  # 3회 이상은 위에서 종료 처리됨

                stt_error_count = 0  # 인식 성공 시 초기화
                silence_error_count = 0  # 여기까지 오면 인식이 성공한거니 다 초기화
                
                ai_res = self.get_gpt_response(user_text)
                print(f"📝 GPT: {ai_res}")
                
                if self.stop_signal: break

                # [결과 처리 로직]
                if "[부정]" in ai_res:
                    self.tts_play(ai_res)
                    if self.callback_result: self.callback_result("DENY")
                    break # 부정은 즉시 종료
                    
                elif "[인정]" in ai_res:
                    self.tts_play(ai_res)
                    
                    # [수정] 대화를 끊지 않고, 점수는 한 번만 반영
                    if self.callback_result and not admitted_once:
                        self.callback_result("ADMIT")
                        admitted_once = True 
                    
                    # break 삭제됨 -> 대화 계속 진행

                elif "[종료]" in ai_res:
                    self.tts_play(ai_res)
                    break 
                else: 
                    self.tts_play(ai_res)

        except Exception as e:
            print(f"Chat Error: {e}")
        finally:
            # [중요] 대화 종료 알림 -> Analyzer가 임계값을 복구함 (4초 -> 2초)
            if self.callback_end: self.callback_end()
            self.is_processing = False
            self.stop_signal = False

    def play_alarm(self):
        if self.is_processing: return
        self.is_processing = True
        self.stop_signal = False 

        try:
            print("\n🚨 [비상] 위험 감지! 비상벨 작동!!!")
            if os.path.exists(self.ALARM_FILE):
                self.play_audio(self.ALARM_FILE, timeout=3.0) 
            self.tts_play("위험합니다! 당장 휴식을 취하세요!")
        except: pass
        finally:
            time.sleep(1)
            self.is_processing = False