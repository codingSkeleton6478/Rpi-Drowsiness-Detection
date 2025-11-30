import os
import time
import pygame
import pyaudio
import wave
import webrtcvad  # [필수] VAD(Voice Activity Detection) 라이브러리: 사람 목소리와 잡음을 구분
from openai import OpenAI

class DriverChatbot:
    """
    [AI 드라이빙 컴패니언 클래스]
    - 역할: 운전자와 음성 대화(STT -> GPT -> TTS)를 수행하고, 대화 내용에서 '졸음 인정/부정' 의도를 파악
    - 특징:
        1. VAD 적용: 차 안의 주행 소음(Wind/Tire Noise)을 무시하고 '사람 목소리'가 들릴 때만 녹음 시작
        2. 환각 필터링: Whisper 모델이 침묵 구간에서 자주 뱉는 "Thanks for watching" 등의 환각 텍스트 제거
        3. 태그 시스템: GPT가 답변에 [인정], [부정] 태그를 달아 시스템(Analyzer)의 상태를 제어
    """
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.is_processing = False 
        
        # [콜백 함수 저장소]
        # 챗봇 내부에서 발생한 이벤트(결과, 대화 시작/종료)를 Main 시스템으로 전달하기 위한 훅(Hook)
        self.callback_result = None  # 결과 처리 (인정/부정) -> Analyzer 리셋 여부 결정
        self.callback_start = None   # 대화 시작 알림 -> Analyzer 임계값 완화 (Speaking Penalty 적용)
        self.callback_end = None     # 대화 종료 알림 -> Analyzer 임계값 복구
        
        self.stop_signal = False 
        
        # [VAD를 위한 오디오 설정]
        # webrtcvad는 16000Hz, 32000Hz, 48000Hz만 지원하며, 음성 인식(Whisper)에는 16000Hz가 가장 효율적임
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  
        self.FRAME_DURATION_MS = 30  # 30ms 단위로 오디오 프레임을 쪼개서 목소리 여부 판별
        self.CHUNK = int(self.RATE * self.FRAME_DURATION_MS / 1000) # 480 프레임 (30ms 분량)
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"
        self.ALARM_FILE = "sounds/alarm.wav"
        
        # [VAD 민감도 설정]
        # 모드 0(민감) ~ 3(둔감). 시끄러운 차내 환경이므로 잡음에 반응하지 않도록 가장 둔감한 '3'으로 설정
        self.vad = webrtcvad.Vad(3)

        # [시스템 프롬프트 설계]
        # GPT에게 페르소나(친구 같은 조수)를 부여하고, 시스템 제어를 위한 '태그([Tag])' 규칙을 주입
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

    # 콜백 연결 함수
    def set_callbacks(self, on_result, on_start, on_end):
        self.callback_result = on_result
        self.callback_start = on_start
        self.callback_end = on_end

    def stop(self):
        """시스템 강제 종료 시 호출 (오디오 재생 중단 및 루프 탈출)"""
        print("🛑 [Chatbot] 강제 정지 명령 수신!")
        self.stop_signal = True
        self.is_processing = False
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except: pass

    def play_audio(self, file_path, timeout=None):
        """오디오 파일 재생 (Non-blocking이지만 필요 시 타임아웃 적용)"""
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
        """
        [TTS: Text to Speech]
        GPT의 텍스트 답변을 음성으로 변환하여 출력.
        - 중요: 출력 시 [태그]는 읽지 않도록 제거(replace) 처리
        """
        if self.stop_signal: return
        
        # 태그 제거 (사용자에게는 "안 잤어! [부정]"에서 [부정]을 들려줄 필요 없음)
        clean_text = text.replace("[대화]", "").replace("[부정]", "").replace("[인정]", "").replace("[종료]", "").strip()
        if not clean_text: return
        try:
            response = self.client.audio.speech.create(model="tts-1", voice="onyx", input=clean_text)
            response.stream_to_file(self.TTS_OUTPUT)
            
            self.messages.append({"role": "assistant", "content": clean_text})
            self.play_audio(self.TTS_OUTPUT)
            
        except Exception as e:
            print(f"TTS Error: {e}")

    # [핵심 기능] VAD 기반 스마트 녹음
    def record_audio(self):
        """
        기존의 '고정 시간 녹음' 방식은 침묵도 녹음하여 API 비용 낭비 및 인식 오류를 유발함.
        -> VAD를 사용하여 '목소리가 들릴 때 시작'하고 '말이 끝나면 종료'하는 스마트 녹음 구현.
        """
        if self.stop_signal: return False
        
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            print("\n🎤 말씀하세요... (목소리가 들리면 녹음을 시작합니다)")
            
            frames = []
            silence_duration = 0      # 말이 끝났는지 판단하기 위한 침묵 지속 시간
            has_spoken = False        # 한 번이라도 말을 했는지 체크
            
            # 최대 10초 대기 (무한 루프 방지)
            start_time = time.time()
            
            while True:
                if self.stop_signal: break
                if time.time() - start_time > 10: # 10초 동안 아무 말 없으면 타임아웃
                    break

                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # VAD 판별: 현재 프레임(30ms)이 사람 목소리인가? (True/False)
                is_speech = self.vad.is_speech(data, self.RATE)
                
                if is_speech:
                    if not has_spoken:
                        print("⚡ 감지됨! 녹음 시작...")
                        has_spoken = True
                    frames.append(data)
                    silence_duration = 0 # 말하는 중이면 침묵 타이머 리셋
                else:
                    # 소리가 안 날 때 (침묵 구간)
                    if has_spoken:
                        # 이미 말을 시작했다면 -> 문장이 끝난 건지 잠깐 쉬는 건지 판단
                        frames.append(data) 
                        silence_duration += self.FRAME_DURATION_MS
                        
                        # 1.0초 이상 조용하면 문장이 끝난 것으로 간주하고 녹음 종료
                        if silence_duration > 1000: 
                            print("✅ 녹음 완료 (문장 끝)")
                            break
                    else:
                        # 아직 말 안 했으면 계속 대기 (버퍼에 담지 않음)
                        pass

            stream.stop_stream()
            stream.close()
            
            if self.stop_signal: return False
            
            # 녹음된 내용이 너무 짧으면(0.5초 미만) 단순 소음(쾅 소리 등)으로 간주하고 버림
            if len(frames) < (self.RATE / self.CHUNK * 0.5):
                return False

            # WAV 파일 저장
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
        """
        [STT: Speech to Text]
        녹음된 오디오를 Whisper API로 전송하여 텍스트로 변환.
        + [환각 필터링]: 조용한 구간에서 Whisper가 '유튜브 자막 데이터'를 뱉는 고질적 문제 해결
        """
        if self.stop_signal: return None
        try:
            if not os.path.exists(self.WAVE_OUTPUT): return None

            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            text = transcript.text.strip()
            
            # [안전장치] Whisper 환각(Hallucination) 필터링 리스트
            # 학습 데이터셋(YouTube)에서 유래한 무의미한 텍스트가 나오면 무시함
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
        """GPT-4o에게 대화 내역 전달 및 응답 생성"""
        if self.stop_signal: return ""
        try:
            self.messages.append({"role": "user", "content": user_text})
            # 토큰 절약을 위해 대화 맥락은 최근 10개(5턴)만 유지
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
        """
        [대화 세션 메인 루프]
        - Analyzer가 졸음을 감지했을 때 호출됨.
        - 운전자에게 말을 걸고, 응답(STT)을 받아 의도를 파악한 뒤 적절한 조치를 취함.
        """
        if self.is_processing: return
        self.is_processing = True
        self.stop_signal = False
        
        # [상태 제어] 한 대화 세션 내에서 '인정' 점수가 중복으로 올라가는 것을 방지
        admitted_once = False 
        # 연속 침묵 및 인식 실패 카운트 (너무 오래 답이 없으면 대화 종료)
        silence_error_count = 0
        stt_error_count = 0
        
        # [중요] 대화 시작 알림 -> Analyzer에게 "지금 말 시키니까 입 벌려도 하품 아님"이라고 통보
        if self.callback_start: self.callback_start()

        try:
            # 대화 맥락 초기화 (새로운 졸음 상황이므로 이전 대화 잊기)
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            # 최대 5번의 턴(Turn) 동안 대화 시도
            for _ in range(5): 
                if self.stop_signal: break

                # 종료 조건 1: 연속 침묵 2회 (운전자가 대화 의지 없음)
                if silence_error_count >= 2:
                    print("  연속 침묵 3회로 대화 종료")
                    self.tts_play("조용하네... 다음에 또 이야기하자!")
                    break
                
                # 종료 조건 2: 연속 인식 실패 3회 (마이크 문제 등)
                if stt_error_count >= 2:
                    print("  연속 인식 실패 3회로 대화 종료")
                    self.tts_play("음성 인식에 문제가 있나봐. 다음에 또 이야기하자!")
                    break
                
                # VAD 녹음 시도
                if not self.record_audio(): 
                    # 말을 안 했거나 잡음만 있었으면 카운트 증가
                    silence_error_count += 1
                    continue

                # 말을 했으면 침묵 카운트 초기화
                silence_error_count = 0
                
                user_text = self.stt()
                if self.stop_signal: break

                print(f"🗣️ 운전자: {user_text}")
                
                # 인식된 텍스트가 유효하지 않을 때
                if not user_text or len(user_text) < 1:
                    stt_error_count += 1
                    print("  음성 인식 실패. 다시 시도해주세요.")

                    if stt_error_count < 3:
                        self.tts_play("잘 못 알아들었어. 다시 말해줄래?")
                        continue
                    else:
                        continue  # 3회 이상은 위에서 종료 처리됨

                stt_error_count = 0  # 인식 성공 시 에러 카운트 초기화
                silence_error_count = 0 
                
                ai_res = self.get_gpt_response(user_text)
                print(f"📝 GPT: {ai_res}")
                
                if self.stop_signal: break

                # [결과 처리 로직: 태그 파싱]
                if "[부정]" in ai_res:
                    self.tts_play(ai_res)
                    if self.callback_result: self.callback_result("DENY") # 시스템 완전 초기화
                    break # 부정은 즉시 종료
                    
                elif "[인정]" in ai_res:
                    self.tts_play(ai_res)
                    
                    # '졸려'라고 인정하면 경고 단계 격상 (단, 세션 당 1회만)
                    if self.callback_result and not admitted_once:
                        self.callback_result("ADMIT")
                        admitted_once = True 
                    
                    # 인정 후에도 잠을 깨우기 위해 대화는 계속 진행 (break 없음)

                elif "[종료]" in ai_res:
                    self.tts_play(ai_res)
                    break # 대화 종료 요청 시 루프 탈출
                else: 
                    self.tts_play(ai_res) # 태그 없으면 일반 대화

        except Exception as e:
            print(f"Chat Error: {e}")
        finally:
            # [중요] 대화 종료 알림 -> Analyzer 임계값 원상 복구
            if self.callback_end: self.callback_end()
            self.is_processing = False
            self.stop_signal = False

    def play_alarm(self):
        """
        [비상벨 작동]
        - 챗봇 로직과는 별개로, 'SLEEP' 단계나 재범 시 강제로 호출되는 최우선 경고
        """
        if self.is_processing: return
        self.is_processing = True
        self.stop_signal = False 

        try:
            print("\n🚨 [비상] 위험 감지! 비상벨 작동!!!")
            # 강력한 알람음 재생 (3초간)
            if os.path.exists(self.ALARM_FILE):
                self.play_audio(self.ALARM_FILE, timeout=3.0) 
            self.tts_play("위험합니다! 당장 휴식을 취하세요!")
        except: pass
        finally:
            time.sleep(1)
            self.is_processing = False