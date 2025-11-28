import os
import time
import pygame
import pyaudio
import wave
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
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"
        self.ALARM_FILE = "sounds/alarm.wav"
        
        # [최종 프롬프트] 의도 파악을 위해 태그를 [부정]과 [인정]으로 분리
        self.system_prompt = """
        너는 졸음운전 방지 AI 조수야. 친구처럼 반말을 써.
        
        [판단 기준]
        1. 운전자가 "괜찮아", "안 잤어", "멀쩡해", "깨어있어" 등 졸음을 **부정**하면 답변 끝에 [부정] 태그를 붙여.
        2. 운전자가 "졸려", "깜빡했네", "미안", "어..." 등 졸음을 **인정**하거나 대답을 흐리면 답변 끝에 [인정] 태그를 붙여.
        3. 운전자가 "종료", "꺼줘"라고 하면 [종료] 태그를 붙여.
        
        [예시]
        유저: "나 안 잤어!" -> "그래? 다행이다. 그래도 조심해! [부정]"
        유저: "아 좀 졸리네..." -> "위험해! 잠깐 쉬어가자. [인정]"
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

    def record_audio(self):
        if self.stop_signal: return False
        
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            print("\n🎤 듣고 있습니다... (말씀하세요)")
            frames = []
            
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if self.stop_signal: break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            if self.stop_signal: return False

            wf = wave.open(self.WAVE_OUTPUT, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        except: return False
        finally: p.terminate()

    def stt(self):
        if self.stop_signal: return None
        try:
            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcript.text
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
        
        # [중요] 대화 시작 알림 -> Analyzer가 임계값을 완화함 (2초 -> 4초)
        if self.callback_start: self.callback_start()

        try:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            for _ in range(3): 
                if self.stop_signal: break
                
                if not self.record_audio(): break
                
                user_text = self.stt()
                if self.stop_signal: break

                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 1:
                    self.tts_play("응? 다시 말해줄래?")
                    continue

                ai_res = self.get_gpt_response(user_text)
                print(f"📝 GPT: {ai_res}")
                
                if self.stop_signal: break

                # [결과 처리 로직]
                if "[부정]" in ai_res:
                    self.tts_play(ai_res)
                    if self.callback_result: self.callback_result("DENY")
                    break 
                elif "[인정]" in ai_res:
                    self.tts_play(ai_res)
                    if self.callback_result: self.callback_result("ADMIT")
                    break
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