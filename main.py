import cv2
import time
import threading
import os
from dotenv import load_dotenv
from picamera2 import Picamera2
from src.detector import DriverMonitor
from src.chatbot import DriverChatbot # [복구]
import src.utils as utils 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ [Error] .env 파일을 찾을 수 없거나 API 키가 없습니다!")
    exit()

def main():
    print("Initializing System (Final AI Mode)...")
    
    monitor = DriverMonitor()
    chatbot = DriverChatbot(OPENAI_API_KEY) # [복구]

    # ==========================================
    # [핵심] 챗봇이 누를 수 있는 '리셋 버튼' 만들기
    # ==========================================
    def trigger_recalibration():
        monitor.is_calibrating = True
        monitor.calib_ear_list = []
        monitor.counter_ear = 0
        # 화면에 표시되게 캘리브레이션 모드로 즉시 전환됨

    # 챗봇에게 버튼 쥐여주기
    chatbot.set_recalibration_callback(trigger_recalibration)
    # ==========================================

    print("Starting Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("Running... 'q' to quit, 'r' to recalibrate manually.")

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 순정 모드 (Dlib 최적화 상태)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            if monitor.is_calibrating:
                monitor.calibrate(frame_bgr, gray)
                
            else:
                status = monitor.process_frame(frame_bgr, gray)
                
                # 졸음 발생 시 -> 챗봇 호출
                if status in ["SLEEP", "YAWN", "HEAD_DOWN"]:
                    print(f"⚠️ 위험 감지: {status}! 챗봇 가동...")
                    
                    # 이미 말하고 있지 않을 때만 실행
                    if not chatbot.is_speaking:
                        t = threading.Thread(target=chatbot.wake_up_driver)
                        t.start()

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Driver Monitor", frame_bgr)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'): # 수동 리셋도 유지
                print("🔄 수동 리셋")
                trigger_recalibration()

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()