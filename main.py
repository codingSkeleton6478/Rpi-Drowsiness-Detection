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
from src.database import DrivingLogDB  # [신규 기능] DB 모듈 추가

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("🚀 Initializing Driver Monitor System (Final Version)...")
    
    detector = FaceDetector()
    analyzer = DriverAnalyzer()
    chatbot = DriverChatbot(OPENAI_API_KEY)
    
    # [신규 기능] DB 객체 초기화
    db = DrivingLogDB()
    last_drowsiness_count = 0  # DB 기록용 카운트 트래커

    # ==============================================================
    # 1. Chatbot <-> Analyzer 연결 (콜백 함수 정의)
    # ==============================================================
    
    # (A) 대화 의도 파악 결과 처리
    def handle_chat_result(result_type):
        if result_type == "DENY": # "안 잤어" -> 완전 초기화
            analyzer.reset_full_calibration()
            
            # [필수 수정] Analyzer가 0이 되었으니, 여기서 기억하는 변수도 0으로 맞춰야 함
            # 이걸 안 하면 다음 졸음 때 (1 > 1)이 False가 되어 기록이 안 남음
            nonlocal last_drowsiness_count
            last_drowsiness_count = 0
            
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
    
    # [신규] 대화 중 수면 유예 타이머 변수 (While 문 밖에서 초기화)
    sleep_grace_start_time = None

    # [신규 기능] 주기적 운전 리포트 스레드 (30분마다)
    def periodic_report_loop():
        while True:
            time.sleep(1800)  # 30분 대기
            
            # 현재 챗봇이나 알람이 작동 중인지 확인 (방해 금지)
            is_busy = (chat_thread is not None and chat_thread.is_alive()) or \
                      (alarm_thread is not None and alarm_thread.is_alive())
            
            if not is_busy:
                # 최근 30분간 졸음 횟수 조회
                count = db.get_count_last_minutes(30)
                if count > 0:
                    msg = f"운전자님, 지난 30분 동안 졸음이 {count}번 감지되었습니다. 휴식을 취하시는 걸 권장드립니다."
                    print(f"📊 [Report] {msg}")
                    chatbot.tts_play(msg)
            else:
                print("📊 [Report] Skipped (System Busy)")

    # 리포트 스레드 시작 (Daemon=True로 설정하여 메인 종료 시 자동 종료)
    report_thread = threading.Thread(target=periodic_report_loop, daemon=True)
    report_thread.start()

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
                
                # [신규 기능] 졸음 카운트 증가 감지 시 DB 자동 기록
                if analyzer.drowsiness_count > last_drowsiness_count:
                    db.log_event("SLEEP")
                    print(f"💾 [DB] 졸음 이벤트 기록됨 (Total: {analyzer.drowsiness_count})")
                    last_drowsiness_count = analyzer.drowsiness_count

                # =========================================================
                # 2. 최종 우선순위 로직 (1초 유예 타임아웃 적용)
                # =========================================================
                
                is_chatting = (chat_thread is not None and chat_thread.is_alive())
                is_alarming = (alarm_thread is not None and alarm_thread.is_alive())

                # [우선순위 1] 비상벨 작동 중이면 간섭 금지
                if is_alarming:
                    sleep_grace_start_time = None # 비상벨 울리면 타이머 초기화
                    pass 

                # [우선순위 2] 챗봇 대화 중일 때
                elif is_chatting:
                    if status == "SLEEP":
                        # Case 1: 초범(Count 0) -> 1초 타이머 유예 적용
                        if analyzer.drowsiness_count == 0:
                            
                            # 1. 타이머가 안 켜져 있으면 지금부터 잰다 (Start Timer)
                            if sleep_grace_start_time is None:
                                sleep_grace_start_time = time.time()
                                cv2.putText(frame_bgr, "Grace Period (Monitoring...)", (10, 150), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 2. 타이머가 켜져 있으면 시간 체크
                            else:
                                elapsed = time.time() - sleep_grace_start_time
                                
                                # [핵심] 1초 이상 지났는데도 계속 SLEEP이면 -> 진짜 자는 것!
                                if elapsed > 1.0:
                                    print(f"🚨 [Emergency] 대화 중 1초 이상 수면 지속! 비상벨!")
                                    chatbot.stop()
                                    alarm_thread = threading.Thread(target=chatbot.play_alarm)
                                    alarm_thread.daemon = True
                                    alarm_thread.start()
                                    analyzer.reset_soft_for_next_stage()
                                    sleep_grace_start_time = None # 타이머 리셋
                                
                                # 아직 1초 안 지났으면 -> 봐줌
                                else:
                                    remaining = 1.0 - elapsed
                                    msg = f"Grace Period ({remaining:.1f}s left)"
                                    cv2.putText(frame_bgr, msg, (10, 150), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Case 2: 재범(Count >= 1) -> 즉시 처형 (타이머 필요 없음)
                        else:
                            print(f"🚨 [Emergency] 재범 감지! 즉시 비상벨!")
                            chatbot.stop() 
                            alarm_thread = threading.Thread(target=chatbot.play_alarm)
                            alarm_thread.daemon = True
                            alarm_thread.start()
                            analyzer.reset_soft_for_next_stage()
                            sleep_grace_start_time = None

                    # SLEEP이 아님 (눈을 떴거나, 그냥 졸린 정도) -> 타이머 리셋
                    else:
                        sleep_grace_start_time = None # 눈 떴으니 타이머 해제
                        
                        if status == "DROWSY":
                             cv2.putText(frame_bgr, "Chatting... (Drowsy)", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                             cv2.putText(frame_bgr, "Chatting...", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # [우선순위 3] 일반 모니터링 상태
                else:
                    sleep_grace_start_time = None # 대화 중 아니면 타이머 쓸 일 없음
                    
                    if status == "SLEEP":
                        print(f"🚨 [Emergency] SLEEP 감지! 비상벨 작동!")
                        alarm_thread = threading.Thread(target=chatbot.play_alarm)
                        alarm_thread.daemon = True
                        alarm_thread.start()
                        analyzer.reset_soft_for_next_stage()

                    elif status == "DROWSY":
                        print(f"⚠️ [Warning] DROWSY 감지 -> 챗봇 대화 시도")
                        chat_thread = threading.Thread(target=chatbot.start_conversation)
                        chat_thread.daemon = True
                        chat_thread.start()

            # FPS 표시
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