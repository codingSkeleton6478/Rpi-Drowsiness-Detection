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
        self.recalibration_callback = None 
        
        # [수정] 강제 종료를 위한 신호 플래그
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
        
        self.system_prompt = """
        너는 졸음운전 방지 AI 조수야. 친구처럼 자연스러운 '반말'을 써.
        운전자가 잠을 깨도록 흥미로운 주제로 대화를 유도하거나 퀴즈를 내줘.
        답변은 1~2문장으로 짧게 해.
        답변 앞에 태그를 붙여: [재설정], [종료], [대화]
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"❌ Audio Init Error: {e}")

    def set_recalibration_callback(self, callback_func):
        self.recalibration_callback = callback_func

    # [신규 기능] 강제 종료 (Kill Switch)
    def stop(self):
        """
        진행 중인 모든 대화와 오디오를 즉시 중단합니다.
        """
        print("🛑 [Chatbot] 강제 정지 명령 수신!")
        self.stop_signal = True  # 정지 신호 발생
        self.is_processing = False
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop() # 오디오 즉시 끄기
        except: pass

    def play_audio(self, file_path, timeout=None):
        if not os.path.exists(file_path): return
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # [수정] 재생 중에도 stop_signal을 계속 감시함
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if self.stop_signal: # 비상 정지 신호 확인
                    pygame.mixer.music.stop()
                    return
                
                if timeout and (time.time() - start_time > timeout):
                    pygame.mixer.music.stop()
                    return
                
                pygame.time.Clock().tick(10) # CPU 과부하 방지
        except: pass

    def tts_play(self, text):
        if self.stop_signal: return # 이미 정지 신호면 실행 안 함
        
        clean_text = text.replace("[대화]", "").replace("[재설정]", "").replace("[종료]", "").strip()
        if not clean_text: return
        try:
            response = self.client.audio.speech.create(model="tts-1", voice="onyx", input=clean_text)
            response.stream_to_file(self.TTS_OUTPUT)
            
            self.messages.append({"role": "assistant", "content": clean_text})
            self.play_audio(self.TTS_OUTPUT) # 여기서도 내부적으로 stop 감시함
            
        except Exception as e:
            print(f"TTS Error: {e}")

    def record_audio(self):
        if self.stop_signal: return False # 정지 신호면 녹음 안 함
        
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            print("\n🎤 듣고 있습니다... (말씀하세요)")
            frames = []
            
            # 녹음 루프 (약 4초)
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                if self.stop_signal: # 녹음 도중 비상 정지
                    print("🛑 녹음 중단됨")
                    break
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            if self.stop_signal: return False # 중단되었으면 저장 안 함

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
        self.stop_signal = False # 시작할 땐 정지 신호 초기화
        
        try:
            # 대화 시작 멘트
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            for _ in range(5): 
                if self.stop_signal: break # 루프 돌 때마다 체크
                
                if not self.record_audio(): break
                
                user_text = self.stt()
                if self.stop_signal: break # STT 중에도 체크

                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 2:
                    self.tts_play("잘 안 들려. 다시 말해줄래?")
                    continue

                ai_res = self.get_gpt_response(user_text)
                print(f"📝 GPT: {ai_res}")
                
                if self.stop_signal: break

                if "[재설정]" in ai_res:
                    self.tts_play(ai_res)
                    if self.recalibration_callback: self.recalibration_callback() 
                    break 
                elif "[종료]" in ai_res:
                    self.tts_play(ai_res)
                    break 
                else: 
                    self.tts_play(ai_res)

        except Exception as e:
            print(f"Chat Error: {e}")
        finally:
            self.is_processing = False
            self.stop_signal = False # 종료 시 플래그 복구

    def play_alarm(self):
        """비상벨은 강제 종료되지 않고 끝까지 울려야 함 (보통)"""
        if self.is_processing: return
        self.is_processing = True
        self.stop_signal = False 

        try:
            print("\n🚨 [비상] 위험 감지! 비상벨 작동!!!")
            if os.path.exists(self.ALARM_FILE):
                # 비상벨은 timeout 2.5초 동안 무조건 울림
                self.play_audio(self.ALARM_FILE, timeout=2.5) 
            
            self.tts_play("위험합니다! 당장 휴식을 취하세요!")
        except: pass
        finally:
            time.sleep(1)
            self.is_processing = False