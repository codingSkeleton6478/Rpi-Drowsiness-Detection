import cv2
import time
import threading
import os
from dotenv import load_dotenv
from picamera2 import Picamera2

# [사용자 정의 모듈 임포트]
# - detector: 얼굴 및 랜드마크 검출 (Dlib)
# - analyzer: 졸음 판단 알고리즘 및 상태 관리 (EAR, MAR, Head Pose)
# - chatbot: OpenAI API 기반 대화 및 음성 처리 (STT/TTS/VAD)
# - database: 졸음 이벤트 로그 저장 (SQLite)
from src.detector import FaceDetector
from src.analyzer import DriverAnalyzer
from src.chatbot import DriverChatbot
from src.database import DrivingLogDB

# 환경 변수 로드 (OpenAI API 키 등 민감 정보 보호)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    """
    [졸음 운전 감지 시스템 메인 실행 함수]
    - 역할: 카메라 입력 처리, 각 모듈(분석, 챗봇, DB) 통합, 우선순위 기반 개입 로직 실행
    - 구조: 무한 루프 내에서 프레임 단위 분석 -> 상태 판단 -> 스레드(경고/대화) 실행
    """
    print("🚀 Initializing Driver Monitor System (Final Version)...")
    
    # 1. 핵심 모듈 초기화
    detector = FaceDetector()
    analyzer = DriverAnalyzer()
    chatbot = DriverChatbot(OPENAI_API_KEY)
    
    # [DB 초기화] 졸음 발생 이력을 로컬 DB에 저장하여 통계 및 리포트에 활용
    db = DrivingLogDB()
    last_drowsiness_count = 0  # DB 중복 기록 방지를 위한 상태 트래커 (Rising Edge 감지용)

    # ==============================================================
    # 1. Chatbot <-> Analyzer 상호작용 콜백 정의
    #    (챗봇의 대화 결과가 분석 로직에 영향을 주도록 설계됨)
    # ==============================================================
    
    # (A) 대화 의도 파악 결과 처리 (챗봇 -> 분석기)
    # - 의도: 운전자가 졸음을 '부정'하면 오탐지로 간주하여 시스템 초기화, '인정'하면 경고 단계 격상
    def handle_chat_result(result_type):
        if result_type == "DENY": 
            # Case 1: "안 잤어" (부정)
            # -> 분석기의 모든 피로도 누적 점수와 카운트를 0으로 초기화 (Calibration 재시도 등)
            analyzer.reset_full_calibration()
            
            # [상태 동기화] Analyzer 카운트가 0이 되었으므로, 메인 루프의 트래커도 0으로 맞춰야
            # 다음 졸음 발생 시 변화량(Diff)을 감지하여 DB에 기록할 수 있음.
            nonlocal last_drowsiness_count
            last_drowsiness_count = 0
            
        elif result_type == "ADMIT": 
            # Case 2: "졸려" (인정)
            # -> 현재까지의 점수는 비우되, 졸음 카운트(drowsiness_count)는 증가시켜 경고 수위를 높임
            analyzer.reset_soft_for_next_stage()

    # (B) 대화 시작 알림 (챗봇 -> 분석기)
    # - 의도: 운전자가 말을 할 때는 입모양이나 눈 깜빡임이 평소와 다르므로, 졸음 판단 임계값을 완화하여 오경보 방지
    def start_speaking_mode():
        analyzer.is_speaking = True
        print("🗣️ [System] Conversation Start -> Threshold Relaxed (4s)")

    # (C) 대화 종료 알림 (챗봇 -> 분석기)
    # - 의도: 대화가 끝나면 다시 엄격한 기준으로 운전자를 감시하기 위해 임계값 복구
    def end_speaking_mode():
        analyzer.is_speaking = False
        print("🤐 [System] Conversation End -> Threshold Normal (2s)")

    # 정의한 콜백 함수들을 챗봇 인스턴스에 등록
    chatbot.set_callbacks(handle_chat_result, start_speaking_mode, end_speaking_mode)

    # 스레드 상태 관리 변수 (Non-blocking 실행을 위해 스레드 사용)
    chat_thread = None
    alarm_thread = None
    
    # [유예 타이머] 대화 중 '눈 감김'을 즉시 수면으로 판정하지 않고, 
    # '말하면서 눈을 감는 행위'인지 '진짜 잠든 것'인지 구분하기 위한 1초 버퍼 변수
    sleep_grace_start_time = None

    # [리포트 스레드] 주기적 통계 알림 기능
    # - 의도: 영상 처리에 부하를 주지 않기 위해 별도 스레드에서 30분마다 DB를 조회하여 운전 리포트 제공
    def periodic_report_loop():
        while True:
            time.sleep(1800)  # 30분 대기 (Blocking call이지만 별도 스레드라 메인 루프 영향 없음)
            
            # 현재 시스템이 대화나 알람으로 바쁜지 확인 (방해 금지 모드)
            is_busy = (chat_thread is not None and chat_thread.is_alive()) or \
                      (alarm_thread is not None and alarm_thread.is_alive())
            
            if not is_busy:
                # 최근 30분 데이터 조회
                count = db.get_count_last_minutes(30)
                if count > 0:
                    msg = f"운전자님, 지난 30분 동안 졸음이 {count}번 감지되었습니다. 휴식을 취하시는 걸 권장드립니다."
                    print(f"📊 [Report] {msg}")
                    chatbot.tts_play(msg)
            else:
                print("📊 [Report] Skipped (System Busy)")

    # 리포트 스레드 시작 (Daemon=True: 메인 프로그램 종료 시 함께 강제 종료)
    report_thread = threading.Thread(target=periodic_report_loop, daemon=True)
    report_thread.start()

    # 2. 라즈베리파이 카메라 초기화 (Picamera2 라이브러리 사용)
    print("📷 Starting Camera...")
    picam2 = Picamera2()
    # 미리보기 구성 (해상도 640x480으로 설정하여 처리 속도 최적화)
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    prev_time = 0
    print("✅ System Ready! Press 'q' to quit, 'r' to recalibrate.")

    try:
        # [메인 루프] 프레임 단위 처리 시작
        while True:
            # (1) 이미지 캡처 및 전처리
            frame = picam2.capture_array()  # Raw 데이터 캡처
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # OpenCV 형식(BGR) 변환
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) # 랜드마크 검출용 흑백 변환

            # (2) 얼굴 및 랜드마크 검출
            shape, face_rect = detector.detect(frame_bgr, gray)

            # (3) 상태 분석 (Analyzer)
            if analyzer.is_calibrating:
                # 초기 실행 시: 사용자 눈/입 크기 기준값 측정
                analyzer.calibrate(frame_bgr, shape, face_rect)
                status = "CALIBRATING"
            else:
                # 일반 실행 시: 졸음 상태 판별
                status = analyzer.process(frame_bgr, shape, face_rect)
                
                # [DB 기록 로직]
                # Analyzer 내부의 졸음 카운트가 증가했는지 확인 (Rising Edge Detection)
                # 매 프레임 기록하는 것을 방지하고, '사건' 발생 시점에만 1회 기록
                if analyzer.drowsiness_count > last_drowsiness_count:
                    db.log_event("SLEEP")
                    print(f"💾 [DB] 졸음 이벤트 기록됨 (Total: {analyzer.drowsiness_count})")
                    last_drowsiness_count = analyzer.drowsiness_count

                # =========================================================
                # 2. 개입 우선순위 로직 (Priority Logic)
                #    우선순위: 비상벨(Alarm) > 대화(Chat) > 일반 모니터링(Monitor)
                # =========================================================
                
                is_chatting = (chat_thread is not None and chat_thread.is_alive())
                is_alarming = (alarm_thread is not None and alarm_thread.is_alive())

                # [Priority 1] 비상벨 작동 중 (최고 위급 상황)
                if is_alarming:
                    sleep_grace_start_time = None # 비상 상황이므로 유예 타이머 무의미 -> 초기화
                    pass  # 비상벨이 울리는 동안은 추가 개입 없이 알람이 끝나길 대기

                # [Priority 2] 챗봇과 대화 중일 때 (경고 단계)
                elif is_chatting:
                    if status == "SLEEP":
                        # Case A: 초범(Count 0)인 경우 -> 'Grace Period(유예 시간)' 적용
                        # 대화 중에는 고개를 끄덕이거나 눈을 길게 감을 수 있으므로 즉시 비상벨을 울리지 않고 1초간 관찰
                        if analyzer.drowsiness_count == 0:
                            
                            # 타이머 시작 (아직 측정 시작 안 했다면)
                            if sleep_grace_start_time is None:
                                sleep_grace_start_time = time.time()
                                cv2.putText(frame_bgr, "Grace Period (Monitoring...)", (10, 150), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 타이머 진행 중
                            else:
                                elapsed = time.time() - sleep_grace_start_time
                                
                                # [임계점 도달] 1초 이상 지속적으로 SLEEP 상태라면 -> 대화고 뭐고 진짜 자는 것
                                if elapsed > 1.0:
                                    print(f"🚨 [Emergency] 대화 중 1초 이상 수면 지속! 비상벨!")
                                    chatbot.stop() # 챗봇 중단
                                    # 비상벨 스레드 전환
                                    alarm_thread = threading.Thread(target=chatbot.play_alarm)
                                    alarm_thread.daemon = True
                                    alarm_thread.start()
                                    analyzer.reset_soft_for_next_stage() # 경고 단계 격상
                                    sleep_grace_start_time = None 
                                
                                # 아직 1초 미만이면 -> 유예 중 (화면 표시만 함)
                                else:
                                    remaining = 1.0 - elapsed
                                    msg = f"Grace Period ({remaining:.1f}s left)"
                                    cv2.putText(frame_bgr, msg, (10, 150), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Case B: 재범(Count >= 1)인 경우 -> 봐주지 않음 (즉시 비상벨)
                        else:
                            print(f"🚨 [Emergency] 재범 감지! 즉시 비상벨!")
                            chatbot.stop() 
                            alarm_thread = threading.Thread(target=chatbot.play_alarm)
                            alarm_thread.daemon = True
                            alarm_thread.start()
                            analyzer.reset_soft_for_next_stage()
                            sleep_grace_start_time = None

                    # SLEEP이 아님 (눈을 떴거나, 그냥 졸린(DROWSY) 정도)
                    else:
                        sleep_grace_start_time = None # 눈을 떴으므로 유예 타이머 해제
                        
                        # 화면에 현재 대화 상태 표시
                        if status == "DROWSY":
                             cv2.putText(frame_bgr, "Chatting... (Drowsy)", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                             cv2.putText(frame_bgr, "Chatting...", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # [Priority 3] 일반 모니터링 상태 (아무 사건 없음)
                else:
                    sleep_grace_start_time = None # 대화 중 아니므로 타이머 불필요
                    
                    if status == "SLEEP":
                        # 즉시 수면 감지 -> 챗봇 건너뛰고 비상벨 직행
                        print(f"🚨 [Emergency] SLEEP 감지! 비상벨 작동!")
                        alarm_thread = threading.Thread(target=chatbot.play_alarm)
                        alarm_thread.daemon = True
                        alarm_thread.start()
                        analyzer.reset_soft_for_next_stage()

                    elif status == "DROWSY":
                        # 졸음 징후 감지 -> 챗봇 대화 시도 (잠 깨우기)
                        print(f"⚠️ [Warning] DROWSY 감지 -> 챗봇 대화 시도")
                        chat_thread = threading.Thread(target=chatbot.start_conversation)
                        chat_thread.daemon = True
                        chat_thread.start()

            # (4) 디버깅 정보 표시 (FPS)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 결과 화면 출력
            cv2.imshow("Driver Monitor (Pi 5)", frame_bgr)

            # 키 입력 처리 (q: 종료, r: 재보정)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): analyzer.reset_full_calibration()

    except Exception as e:
        print(f"❌ Critical Error: {e}")

    finally:
        # 프로그램 종료 시 자원 해제
        print("🛑 Stopping System...")
        if chatbot: chatbot.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()