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
        self.recalibration_callback = None 
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 4
        
        self.WAVE_OUTPUT = "user_input.wav"
        self.TTS_OUTPUT = "ai_response.mp3"

    def set_recalibration_callback(self, callback_func):
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
        """
        텍스트를 음성으로 변환하여 재생합니다.
        답변 앞에 붙은 태그([대화], [종료] 등)는 제거하고 재생합니다.
        """
        # 태그 제거 (사용자에게는 태그를 들려주지 않음)
        clean_text = text.replace("[대화]", "").replace("[재설정]", "").replace("[종료]", "").strip()
        
        if not clean_text: return # 할 말이 없으면 재생 안 함

        try:
            print(f"🤖 봇(TTS): {clean_text}")
            response = self.client.audio.speech.create(
                model="tts-1", voice="onyx", input=clean_text
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
            
            print("\n🎤 듣고 있습니다... (말씀하세요)")
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

    def get_gpt_response(self, user_text):
        """
        GPT에게 운전자의 말을 전달하고, [태그]와 함께 답변을 받습니다.
        """
        system_prompt = """
        너는 졸음운전 방지 시스템의 AI 조수야. 운전자가 졸지 않게 계속 말을 걸거나 도와줘야 해.
        친구처럼 자연스러운 '반말'을 사용해. 대답은 1~2문장으로 짧게 해.

        너는 운전자의 말을 듣고 **가장 먼저 의도를 파악해서** 답변 맨 앞에 태그를 붙여야 해.

        ---
        **[판단 기준]**
        1. **[재설정]**: 운전자가 "안 졸려", "오류야", "다시 맞춰봐", "잘못 본 거야"라고 시스템을 불신하거나 재설정을 요구할 때.
           -> 답변 예시: "[재설정] 아, 내가 착각했나 봐. 다시 설정할게 정면 봐줘."
           
        2. **[종료]**: 운전자가 "됐어", "조용히 해", "필요 없어", "그만"이라고 하며 **대화를 완전히 끝내고 싶어할 때.**
           (주의: "아니 노래 틀어줘"처럼 거절 뒤에 다른 요청이 있으면 절대 종료하면 안 됨!)
           -> 답변 예시: "[종료] 알겠어. 운전 조심해!"

        3. **[대화]**: 그 외 모든 경우 (일상 대화, 졸리다는 호소, 정보 요청 등).
           특히 **"아니, 휴게소 찾아줘", "필요 없고 노래나 불러줘"** 처럼 앞에는 부정하지만 뒤에 요청이 있는 경우 무조건 [대화]로 분류해.
           -> 답변 예시: "[대화] 그래, 신나는 노래 불러줄게!", "[대화] 가장 가까운 휴게소는 5km 앞에 있어."
        ---
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7 # 창의성 약간 추가 (자연스러운 대화 위해)
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"GPT Error: {e}")
            return "[종료] 미안, 오류가 났어."

    def wake_up_driver(self):
        if self.is_speaking: return
        self.is_speaking = True
        
        try:
            # 1. 최초 경고 멘트
            self.tts_play("운전자님! 깜빡 조신 것 같은데? 괜찮아?")
            
            # 2. 대화 루프 (최대 5턴)
            for _ in range(5): 
                if not self.record_audio(): break
                user_text = self.stt()
                print(f"🗣️ 운전자: {user_text}")
                
                if not user_text or len(user_text) < 2:
                    self.tts_play("잘 안 들려. 다시 말해줄래?")
                    continue

                # 3. GPT 응답 받기
                ai_full_response = self.get_gpt_response(user_text)
                print(f"📝 GPT 원본 응답: {ai_full_response}") # 디버깅용

                # 4. 태그 분석 및 행동
                if "[재설정]" in ai_full_response:
                    self.tts_play(ai_full_response)
                    if self.recalibration_callback:
                        print("🔄 [Chatbot] 시스템 재설정 요청...")
                        self.recalibration_callback() 
                    break 

                elif "[종료]" in ai_full_response:
                    self.tts_play(ai_full_response)
                    break 

                else: 
                    # [대화] 태그이거나 태그가 없는 경우 -> 계속 대화
                    # 만약 태그가 없다면 기본적으로 대화로 간주
                    self.tts_play(ai_full_response)
                    # 루프 계속 돔 (운전자의 다음 말 대기)

        except Exception as e:
            print(f"Chat Process Error: {e}")
        
        finally:
            self.is_speaking = False