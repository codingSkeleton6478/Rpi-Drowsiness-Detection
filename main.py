import cv2
import time
import threading
import numpy as np
import pygame
import os
from dotenv import load_dotenv
from picamera2 import Picamera2
from src.detector import DriverMonitor
import src.utils as utils 

# from src.chatbot import DriverChatbot 

# ==========================================
# 비프음 생성기
# ==========================================
class BeepAlert:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        except Exception as e:
            print(f"Audio Init Error: {e}")
        
        duration = 0.5 
        frequency = 880 
        sample_rate = 44100
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(frequency * t * 2 * np.pi)
        
        audio = (tone * 32767).astype(np.int16)
        self.sound = pygame.sndarray.make_sound(audio)

    def play(self):
        if pygame.mixer.get_init() and not pygame.mixer.get_busy():
            print("🔊 [경고] 비프음 출력!")
            self.sound.play()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("Initializing System (Noise Reduction Mode)...")
    
    monitor = DriverMonitor()
    # chatbot = DriverChatbot(OPENAI_API_KEY)
    beeper = BeepAlert()

    print("Starting Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("Running... Press 'q' to quit, 'r' to recalibrate.")

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # ---------------------------------------------------------
            # [핵심 수정] 노이즈 제거 (Gaussian Blur)
            # ---------------------------------------------------------
            # (5, 5)는 뭉개는 강도입니다. 노이즈가 심하면 (7, 7)로 올려보세요.
            frame_bgr = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
            
            # 부드러워진 이미지를 흑백으로 변환 -> Dlib이 훨씬 좋아합니다.
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            if monitor.is_calibrating:
                monitor.calibrate(frame_bgr, gray)
                
            else:
                status = monitor.process_frame(frame_bgr, gray)
                
                if status in ["SLEEP", "YAWN", "HEAD_DOWN"]:
                    print(f"⚠️ 위험 감지: {status}!")
                    beeper.play()
                    
                    # if not chatbot.is_speaking:
                    #     t = threading.Thread(target=chatbot.wake_up_driver)
                    #     t.start()

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 출력 (약간 뽀샤시해진 화면이 보일 겁니다)
            cv2.imshow("Driver Monitor", frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 리셋: 다시 캘리브레이션 합니다.")
                monitor.is_calibrating = True
                monitor.calib_ear_list = []

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()