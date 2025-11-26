import os
import time
import pygame
import pyaudio
import wave
from openai import OpenAI

class DriverChatbot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.is_speaking = False
        self.recalibration_callback = None # 재설정 기능을 연결할 스위치
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"

    def set_recalibration_callback(self, callback_func):
        """main.py에서 재설정 함수를 받아오는 곳"""
        self.recalibration_callback = callback_func

    def play_audio(self, file_path):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Audio Play Error: {e}")

    def tts_play(self, text):
        try:
            print(f"🤖 봇: {text}")
            response = self.client.audio.speech.create(
                model="tts-1", voice="onyx", input=text
            )
            response.stream_to_file(self.TTS_OUTPUT)
            self.play_audio(self.TTS_OUTPUT)
        except Exception as e:
            print(f"TTS Error: {e}")

    def record_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
            
            print("🎤 듣고 있습니다... (말씀하세요)")
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
        except Exception as e:
            print(f"Mic Error: {e}")
            return False
        finally:
            p.terminate()

    def stt(self):
        try:
            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            return transcript.text
        except Exception as e:
            print(f"STT Error: {e}")
            return None

    def get_gpt_intent_and_response(self, user_text):
        """
        GPT에게 운전자의 의도를 3가지(RESET, STOP, KEEP) 중 하나로 분류하고 답변을 요청
        """
        system_prompt = """
        너는 졸음운전 방지 시스템의 AI 비서야. 운전자의 말을 듣고 다음 3가지를 판단해.
        형식은 반드시 "TAG|답변" 형태로 말해. (예: KEEP|네, 신나는 노래 틀어드릴게요.)

        1. [RESET]: 운전자가 "안 졸려", "오류야", "다시 측정해"라고 하며 시스템 재설정을 원할 때.
           -> 답변: 알겠습니다. 정면을 봐주세요. 다시 설정합니다.
        2. [STOP]: 운전자가 "괜찮아", "아니", "그만", "없어"라고 하며 대화를 끝내고 싶어할 때.
           -> 답변: 네, 알겠습니다. 안전 운전하세요!
        3. [KEEP]: 그 외 모든 일상 대화나 요청 (졸리다, 노래 틀어줘 등).
           -> 답변: (운전자 말에 대한 짧고 친근한 반말 대답)
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )
            content = response.choices[0].message.content
            if "|" in content:
                return content.split("|", 1) # [TAG, Response] 분리
            else:
                return "KEEP", content # 형식 안 지키면 그냥 대화 유지
        except Exception as e:
            print(f"GPT Error: {e}")
            return "STOP", "미안, 에러가 났어."

    # ==========================================
    # [최종] 대화 루프 (재설정 + 꼬리물기)
    # ==========================================
    def wake_up_driver(self):
        if self.is_speaking: return
        self.is_speaking = True
        
        try:
            # 1. 최초 경고
            self.tts_play("운전자님! 졸음이 감지됐어! 괜찮아?")
            
            # 2. 대화 반복 (최대 5턴)
            for _ in range(5): 
                if not self.record_audio(): break
                user_text = self.stt()
                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 2:
                    self.tts_play("잘 안 들려. 다시 말해줄래?")
                    continue

                # 3. GPT가 의도 파악 & 답변 생성
                tag, ai_reply = self.get_gpt_intent_and_response(user_text)
                
                # 상황별 행동
                if tag.strip() == "RESET":
                    self.tts_play(ai_reply)
                    if self.recalibration_callback:
                        print("🔄 [Chatbot] 재설정 명령 실행!")
                        self.recalibration_callback() # main.py의 리셋 함수 호출
                    break # 대화 종료하고 리셋하러 감

                elif tag.strip() == "STOP":
                    self.tts_play(ai_reply)
                    break # 대화 종료

                else: # KEEP (계속 대화)
                    # 답변 후 꼬리 질문 붙이기
                    full_reply = ai_reply + " 혹시 더 필요한 거 있어?"
                    self.tts_play(full_reply)
                    # 루프 다시 돔 (운전자 대답 대기)

        except Exception as e:
            print(f"Chat Process Error: {e}")
        
        finally:
            self.is_speaking = False