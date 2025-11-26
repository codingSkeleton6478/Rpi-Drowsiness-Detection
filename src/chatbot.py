import os
import threading
import pyaudio
import wave
import pygame
from openai import OpenAI

class DriverChatbot:
    def __init__(self, api_key):
        # main.py에서 넘겨받은 키로 OpenAI 연결
        self.client = OpenAI(api_key=api_key)
        self.is_speaking = False  # 중복 실행 방지용 깃발
        
        # 오디오 설정 (아까 테스트한 그 설정 그대로)
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4  # 녹음 시간 (4초)
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"

    def tts_play(self, text):
        """텍스트 -> 음성 재생 (TTS)"""
        try:
            print(f"🔊 [TTS] 음성 생성 중: {text}")
            response = self.client.audio.speech.create(
                model="tts-1", voice="onyx", input=text
            )
            response.stream_to_file(self.TTS_OUTPUT)

            # 재생
            pygame.mixer.init()
            pygame.mixer.music.load(self.TTS_OUTPUT)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"❌ TTS Error: {e}")

    def wake_up_driver(self):
        """
        [메인 기능] 졸음 감지 시 호출됨
        1. 깨우는 멘트 출력
        2. 운전자 말 듣기 (녹음)
        3. GPT와 대화
        """
        if self.is_speaking: 
            return  # 이미 떠들고 있으면 무시

        self.is_speaking = True  # "나 말하는 중이야!" 표시
        
        try:
            # 1. 경고 멘트 (먼저 말을 걺)
            self.tts_play("운전자님! 졸음이 감지되었습니다. 저랑 대화해요!")
            
            # 2. 녹음 시작
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
            
            print("🎤 [STT] 듣고 있습니다... (말씀하세요)")
            frames = []
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 파일 저장
            wf = wave.open(self.WAVE_OUTPUT, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 3. STT (변환)
            audio_file = open(self.WAVE_OUTPUT, "rb")
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            user_text = transcript.text
            print(f"🗣️ 운전자: {user_text}")

            if user_text:
                # 4. GPT (답변)
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "졸음운전 방지 시스템이야. 운전자가 잠을 깨도록 짧고 활기차게 반말로 대답해."},
                        {"role": "user", "content": user_text}
                    ]
                )
                ai_reply = response.choices[0].message.content
                print(f"🤖 봇: {ai_reply}")
                
                # 5. 대답 말하기
                self.tts_play(ai_reply)

        except Exception as e:
            print(f"❌ Chatbot Error: {e}")
        
        finally:
            self.is_speaking = False  # "이제 말 끝났어" 표시