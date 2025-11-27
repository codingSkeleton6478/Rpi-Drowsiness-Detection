import cv2
import time
import threading
import os
from dotenv import load_dotenv
from picamera2 import Picamera2

# 우리가 만든 모듈들 불러오기
from src.detector import FaceDetector
from src.analyzer import DriverAnalyzer
from src.chatbot import DriverChatbot

# .env 파일에서 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    # ==========================================
    # 1. 시스템 초기화 (System Init)
    # ==========================================
    print("🚀 Initializing Driver Monitor System on Raspberry Pi 5...")
    
    detector = FaceDetector()               # 센서
    analyzer = DriverAnalyzer()             # 두뇌
    chatbot = DriverChatbot(OPENAI_API_KEY) # 행동
    
    # 챗봇이 '재설정'을 요청하면 analyzer를 초기화하는 함수 연결
    def trigger_recalibration():
        print("🔄 챗봇 요청으로 재설정 시작")
        analyzer.is_calibrating = True
        analyzer.calib_ear_list = []
        # [Step 1 추가] 입 너비 리스트도 초기화
        if hasattr(analyzer, 'calib_mouth_width_list'):
            analyzer.calib_mouth_width_list = []

    chatbot.set_recalibration_callback(trigger_recalibration)

    # ==========================================
    # 2. 카메라 설정
    # ==========================================
    print("📷 Starting Camera...")
    picam2 = Picamera2()
    # Picamera2 설정 (색상 왜곡 방지용 기본 설정)
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("✅ System Ready! Press 'q' to quit, 'r' to recalibrate.")

    # ==========================================
    # 3. 메인 루프 (Main Loop)
    # ==========================================
    try:
        while True:
            # (1) 프레임 획득
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # (2) 데이터 추출 (Detector)
            # [수정 완료] 이제 올바른 함수 이름인 detect를 호출합니다.
            shape, face_rect = detector.detect(frame_bgr, gray)

            # (3) 상태 판단 (Analyzer)
            if analyzer.is_calibrating:
                # 캘리브레이션 모드: 데이터 수집 (입 너비 포함)
                analyzer.calibrate(frame_bgr, shape, face_rect)
                status = "CALIBRATING"
            else:
                # 주행 모드: 졸음 감시
                status = analyzer.process(frame_bgr, shape, face_rect)

                # (4) 챗봇/비상벨 실행 (Chatbot) - 스레딩
                if not chatbot.is_processing:
                    # 1단계: 졸음/하품 -> 대화
                    if status in ["DROWSY", "YAWN"]:
                        t = threading.Thread(target=chatbot.start_conversation)
                        t.daemon = True 
                        t.start()
                    
                    # 2단계: 수면/고개 숙임 -> 비상벨
                    elif status in ["SLEEP", "HEAD_DOWN"]:
                        t = threading.Thread(target=chatbot.play_alarm)
                        t.daemon = True
                        t.start()

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 화면 출력
            cv2.imshow("Driver Monitor (Pi 5)", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                trigger_recalibration()

    except Exception as e:
        print(f"❌ Critical Error: {e}")

    finally:
        print("🛑 Stopping System...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()