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
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"
        self.ALARM_FILE = "sounds/alarm.wav"
        
        # ======================================================
        # [핵심 수정] 대화 기억 저장소 (Memory)
        # ======================================================
        self.system_prompt = """
        너는 졸음운전 방지 AI 조수야. 친구처럼 자연스러운 '반말'을 써.
        운전자가 잠을 깨도록 흥미로운 주제로 대화를 유도하거나 퀴즈를 내줘.
        
        중요: 답변 맨 앞에 반드시 아래 태그 중 하나를 붙여야 해.
        [재설정]: 운전자가 시스템 오류를 지적하거나 재설정을 원할 때.
        [종료]: 운전자가 대화를 거부하거나 그만하라고 할 때.
        [대화]: 그 외 모든 경우 (퀴즈 정답 확인 포함).
        
        답변은 1~2문장으로 짧게 해.
        """
        self.messages = [{"role": "system", "content": self.system_prompt}]

        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"❌ Audio Init Error: {e}")

    def set_recalibration_callback(self, callback_func):
        self.recalibration_callback = callback_func

    def play_audio(self, file_path, timeout=None):
        if not os.path.exists(file_path): return
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            if timeout:
                time.sleep(timeout)
                pygame.mixer.music.stop()
            else:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
        except: pass

    def tts_play(self, text):
        clean_text = text.replace("[대화]", "").replace("[재설정]", "").replace("[종료]", "").strip()
        if not clean_text: return
        try:
            response = self.client.audio.speech.create(model="tts-1", voice="onyx", input=clean_text)
            response.stream_to_file(self.TTS_OUTPUT)
            self.play_audio(self.TTS_OUTPUT, timeout=None)
            
            # [중요] 봇이 말한 내용도 기억에 추가해야 문맥이 이어짐
            self.messages.append({"role": "assistant", "content": clean_text})
            
        except Exception as e:
            print(f"TTS Error: {e}")

    def record_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
            print("\n🎤 듣고 있습니다...")
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
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
        try:
            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcript.text
        except: return None

    def get_gpt_response(self, user_text):
        """
        대화 히스토리를 포함하여 GPT에게 요청
        """
        try:
            # 1. 사용자 발화 기억 추가
            self.messages.append({"role": "user", "content": user_text})
            
            # 2. 기억이 너무 길어지면(10개 이상) 앞부분 삭제 (토큰 절약)
            if len(self.messages) > 10:
                # system prompt(0번)는 유지하고 그 뒤 오래된 대화 삭제
                self.messages = [self.messages[0]] + self.messages[-9:]

            # 3. 전체 히스토리 전송
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages, # <--- 여기가 핵심
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"GPT Error: {e}")
            return "[종료] 오류가 발생했어."

    def start_conversation(self):
        if self.is_processing: return
        self.is_processing = True
        
        try:
            # 대화 시작 전 히스토리 초기화 (새로운 상황이므로)
            # 단, 졸음 상황이라는 건 알려줘야 함
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "assistant", "content": "운전자님! 깜빡 조신 것 같은데? 괜찮아?"}
            ]
            
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            for _ in range(5): 
                if not self.record_audio(): break
                user_text = self.stt()
                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 2:
                    self.tts_play("잘 안 들려. 다시 말해줄래?")
                    continue

                ai_full_response = self.get_gpt_response(user_text)
                print(f"📝 GPT 응답: {ai_full_response}")

                if "[재설정]" in ai_full_response:
                    self.tts_play(ai_full_response)
                    if self.recalibration_callback: self.recalibration_callback() 
                    break 
                elif "[종료]" in ai_full_response:
                    self.tts_play(ai_full_response)
                    break 
                else: 
                    self.tts_play(ai_full_response)

        except Exception as e:
            print(f"Chat Error: {e}")
        finally:
            self.is_processing = False

    def play_alarm(self):
        if self.is_processing: return
        self.is_processing = True

        try:
            print("\n🚨 [2단계] 위험 감지! 비상벨 작동!!!")
            if os.path.exists(self.ALARM_FILE):
                self.play_audio(self.ALARM_FILE, timeout=2.5)
            else:
                time.sleep(1)

            self.tts_play("위험합니다! 당장 휴식을 취하세요!")
        except: pass
        finally:
            time.sleep(1)
            self.is_processing = False