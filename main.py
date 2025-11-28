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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("🚀 Initializing Driver Monitor System (Final Version)...")
    
    detector = FaceDetector()
    analyzer = DriverAnalyzer()
    chatbot = DriverChatbot(OPENAI_API_KEY)
    
    # ==============================================================
    # 1. Chatbot <-> Analyzer 연결 (콜백 함수 정의)
    # ==============================================================
    
    # (A) 대화 의도 파악 결과 처리
    def handle_chat_result(result_type):
        if result_type == "DENY": # "안 잤어" -> 완전 초기화
            analyzer.reset_full_calibration()
        elif result_type == "ADMIT": # "졸려" -> 카운트 증가 (경고 격상)
            analyzer.reset_soft_for_next_stage()

    # (B) 대화 시작 알림 -> 말하는 모드 ON (임계값 2배 완화)
    def start_speaking_mode():
        analyzer.is_speaking = True
        print("🗣️ [System] Conversation Start -> Threshold Relaxed (4s)")

    # (C) 대화 종료 알림 -> 말하는 모드 OFF (임계값 정상화)
    def end_speaking_mode():
        analyzer.is_speaking = False
        print("🤐 [System] Conversation End -> Threshold Normal (2s)")

    # 챗봇에 콜백 등록
    chatbot.set_callbacks(handle_chat_result, start_speaking_mode, end_speaking_mode)

    # 스레드 변수
    chat_thread = None
    alarm_thread = None

    print("📷 Starting Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("✅ System Ready! Press 'q' to quit, 'r' to recalibrate.")

    try:
        while True:
            # (1) 프레임 처리
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            shape, face_rect = detector.detect(frame_bgr, gray)

            if analyzer.is_calibrating:
                analyzer.calibrate(frame_bgr, shape, face_rect)
                status = "CALIBRATING"
            else:
                status = analyzer.process(frame_bgr, shape, face_rect)

                # =========================================================
                # 2. 최종 우선순위 로직 (Drowsiness Count 반영)
                # =========================================================
                
                is_chatting = (chat_thread and chat_thread.is_alive())
                is_alarming = (alarm_thread and alarm_thread.is_alive())

                # [우선순위 1] 비상벨 작동 중이면 상태 유지
                if is_alarming:
                    pass 

                # [우선순위 2] 챗봇 대화 중일 때
                elif is_chatting:
                    if status == "SLEEP":
                        # Case 1: 초범(Count 0) -> 봐줌 (유예)
                        if analyzer.drowsiness_count == 0:
                            cv2.putText(frame_bgr, "Grace Period (Chatting)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Case 2: 재범(Count >= 1) -> 처형 (챗봇 중단 후 비상벨)
                        else:
                            print(f"🚨 [Emergency] 재범 감지! 챗봇 중단 후 비상벨 작동!")
                            chatbot.stop() 
                            alarm_thread = threading.Thread(target=chatbot.play_alarm)
                            alarm_thread.daemon = True
                            alarm_thread.start()
                    else:
                        cv2.putText(frame_bgr, "Chatting...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # [우선순위 3] 일반 모니터링 상태
                else:
                    if status == "SLEEP":
                        print(f"🚨 [Emergency] {status} 감지! 비상벨 작동!")
                        alarm_thread = threading.Thread(target=chatbot.play_alarm)
                        alarm_thread.daemon = True
                        alarm_thread.start()

                    # YAWN은 챗봇 트리거에서 제외됨. 오직 DROWSY만.
                    elif status == "DROWSY":
                        print(f"⚠️ [Warning] {status} 감지 -> 챗봇 대화 시도")
                        chat_thread = threading.Thread(target=chatbot.start_conversation)
                        chat_thread.daemon = True
                        chat_thread.start()

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Driver Monitor (Pi 5)", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): analyzer.reset_full_calibration()

    except Exception as e:
        print(f"❌ Critical Error: {e}")

    finally:
        print("🛑 Stopping System...")
        if chatbot: chatbot.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()