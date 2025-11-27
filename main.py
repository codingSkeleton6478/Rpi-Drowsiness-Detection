import cv2
import time
import threading
import os
from dotenv import load_dotenv
from picamera2 import Picamera2

# 모듈 불러오기
from src.detector import FaceDetector
from src.analyzer import DriverAnalyzer
from src.chatbot import DriverChatbot

# .env 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    # ==========================================
    # 1. 시스템 초기화
    # ==========================================
    print("🚀 Initializing Driver Monitor System on Raspberry Pi 5...")
    
    detector = FaceDetector()
    analyzer = DriverAnalyzer()
    chatbot = DriverChatbot(OPENAI_API_KEY)
    
    # 챗봇 재설정 콜백
    def trigger_recalibration():
        print("🔄 챗봇 요청으로 재설정 시작")
        analyzer.is_calibrating = True
        analyzer.calib_ear_list = []
        if hasattr(analyzer, 'calib_mouth_width_list'):
            analyzer.calib_mouth_width_list = []

    chatbot.set_recalibration_callback(trigger_recalibration)

    # ==========================================
    # 2. 스레드 상태 관리 변수 (핵심)
    # ==========================================
    # 현재 실행 중인 스레드를 추적하여 중복 실행 방지 및 우선순위 제어
    chat_thread = None
    alarm_thread = None

    # ==========================================
    # 3. 카메라 설정
    # ==========================================
    print("📷 Starting Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("✅ System Ready! Press 'q' to quit, 'r' to recalibrate.")

    # ==========================================
    # 4. 메인 루프
    # ==========================================
    try:
        while True:
            # (1) 프레임 획득
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # (2) 데이터 추출
            shape, face_rect = detector.detect(frame_bgr, gray)

            # (3) 상태 판단
            if analyzer.is_calibrating:
                analyzer.calibrate(frame_bgr, shape, face_rect)
                status = "CALIBRATING"
            else:
                status = analyzer.process(frame_bgr, shape, face_rect)

                # =========================================================
                # [최종 로직] 우선순위 기반 행동 제어 (Priority Control)
                # =========================================================
                
                # [1순위] 위험 상황 (SLEEP, HEAD_DOWN) -> 즉시 개입
                if status in ["SLEEP", "HEAD_DOWN"]:
                    # 이미 비상벨이 울리고 있다면 건드리지 않음
                    if alarm_thread and alarm_thread.is_alive():
                        pass
                    else:
                        print(f"🚨 [Emergency] {status} 감지! 챗봇 중단 후 비상벨 작동!")
                        
                        # 1. 수다 떨던 챗봇 강제 종료 (Interrupt)
                        chatbot.stop()
                        
                        # 2. 비상벨 스레드 시작
                        alarm_thread = threading.Thread(target=chatbot.play_alarm)
                        alarm_thread.daemon = True
                        alarm_thread.start()

                # [2순위] 경고 상황 (DROWSY, YAWN) -> 챗봇 대화
                elif status in ["DROWSY", "YAWN"]:
                    # 비상벨이 울리는 중이면 챗봇 실행 안 함 (비상벨이 우선)
                    is_alarm_running = (alarm_thread and alarm_thread.is_alive())
                    
                    # 이미 채팅 중이면 냅둠
                    is_chat_running = (chat_thread and chat_thread.is_alive())

                    if not is_alarm_running and not is_chat_running:
                        print(f"⚠️ [Warning] {status} 감지 -> 챗봇 대화 시도")
                        chat_thread = threading.Thread(target=chatbot.start_conversation)
                        chat_thread.daemon = True
                        chat_thread.start()

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
        # 종료 시 스레드 정리
        if chatbot: chatbot.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()