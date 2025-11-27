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

if not OPENAI_API_KEY:
    print("❌ [Error] .env 파일을 찾을 수 없거나 OPENAI_API_KEY가 없습니다!")
    exit()

def main():
    # ==========================================
    # 1. 시스템 초기화 (System Init)
    # ==========================================
    print("🚀 Initializing Driver Monitor System on Raspberry Pi 5...")
    
    detector = FaceDetector()               # 센서
    analyzer = DriverAnalyzer()             # 두뇌
    chatbot = DriverChatbot(OPENAI_API_KEY) # 행동

    # ==========================================
    # [수정] 카메라 설정 (사용자 원본 설정으로 복구)
    # 불필요한 'format="XRGB8888"'을 제거하여 색상 왜곡 해결
    # ==========================================
    print("📷 Starting Camera...")
    picam2 = Picamera2()
    # 원본처럼 size만 지정 (기본값이 3채널 RGB라 변환하기 가장 좋음)
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    
    print("✅ System Ready! Press 'q' to quit, 'r' to recalibrate.")

    # ==========================================
    # 2. 메인 루프 (Main Loop)
    # ==========================================
    try:
        while True:
            # (1) 프레임 획득
            frame = picam2.capture_array()
            
            # [색상 보정] RGB -> BGR 변환 (이게 있어야 얼굴이 파랗게 안 나옴)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 흑백 변환 (얼굴 인식용)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # (2) 데이터 추출 (Detector)
            face_data = detector.process_frame(frame_bgr, gray)

            # (3) 상태 판단 (Analyzer)
            status, info = analyzer.process(face_data)

            # (4) 챗봇/비상벨 실행 (Chatbot) - 스레딩
            if not chatbot.is_processing:
                
                # 1단계: 졸음/하품 -> 대화 (Soft)
                if status in ["DROWSY", "YAWN"]:
                    print(f"⚠️ [Main] 감지됨: {status} -> 챗봇 스레드 시작")
                    t = threading.Thread(target=chatbot.start_conversation)
                    t.daemon = True 
                    t.start()

                # 2단계: 수면/고개 숙임 -> 비상벨 (Hard)
                elif status in ["CRITICAL_SLEEP", "HEAD_DOWN"]:
                    print(f"🚨 [Main] 위험 감지: {status} -> 비상벨 스레드 시작")
                    t = threading.Thread(target=chatbot.play_alarm)
                    t.daemon = True
                    t.start()

            # (5) 화면 정보 표시
            if status == "CALIBRATING":
                prog = info.get('progress', 0)
                total = info.get('total', 50)
                cv2.putText(frame_bgr, f"CALIBRATING... {prog}/{total}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            elif status == "NO_FACE":
                cv2.putText(frame_bgr, "WAITING FOR DRIVER...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
            else:
                ear = info.get('ear', 0)
                thresh = info.get('ear_thresh', 0)
                mar = info.get('mar', 0)
                
                color = (0, 255, 0) # Green
                if status in ["DROWSY", "YAWN"]: color = (0, 165, 255) # Orange
                elif status in ["CRITICAL_SLEEP", "HEAD_DOWN"]: color = (0, 0, 255) # Red

                cv2.putText(frame_bgr, f"STATUS: {status}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.putText(frame_bgr, f"EAR: {ear:.2f} (Thresh: {thresh:.2f})", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
                print("🔄 재설정 요청")
                analyzer.is_calibrating = True
                analyzer.calib_ear_list = []

    except Exception as e:
        print(f"❌ Critical Error: {e}")

    finally:
        print("🛑 Stopping System...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()