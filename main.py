import cv2
import time
import threading
from picamera2 import Picamera2
from src.detector import DriverMonitor
from src.chatbot import DriverChatbot

# ==========================================
# API 키 설정
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# ==========================================

def main():
    # 1. 객체 생성
    print("Initializing System...")
    monitor = DriverMonitor()
    chatbot = DriverChatbot(OPENAI_API_KEY)

    # 2. 카메라 설정 (PiCamera2)
    print("Starting Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("Running... Press 'q' to quit.")

    try:
        while True:
            # 3. 프레임 캡처
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # 4. 얼굴 감지
            faces = monitor.detector(gray, 0)
            
            if monitor.is_calibrating:
                # 초기 캘리브레이션 모드
                monitor.calibrate(frame_bgr, faces, gray)
            else:
                # 감시 모드 실행 -> 현재 상태(status) 받아옴
                status = monitor.process_frame(frame_bgr, faces, gray)
                
                # 5. 졸음 발생 시 챗봇 실행 (이미 말하고 있으면 실행 안 함)
                if status == "SLEEP" and not chatbot.is_speaking:
                    print("⚠️ Drowsiness Detected! Activating Chatbot...")
                    # 백그라운드 스레드로 실행 (화면 멈춤 방지)
                    t = threading.Thread(target=chatbot.wake_up_driver)
                    t.start()

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 6. 화면 출력
            cv2.imshow("Driver Monitor", frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("System Stopped.")

if __name__ == "__main__":
    main()