import cv2
import time
import numpy as np
from picamera2 import Picamera2

# 모듈 불러오기 (챗봇 제외)
from src.detector import FaceDetector
from src.analyzer import DriverAnalyzer
import src.utils as utils

def main():
    print("🔧 [TEST MODE] Starting Driver Monitor without AI Chatbot...")
    
    # 1. 객체 생성
    detector = FaceDetector()
    analyzer = DriverAnalyzer()
    
    # 2. 카메라 설정
    print("📷 Starting Camera...")
    picam2 = Picamera2()
    # 색상 왜곡 방지용 사이즈 설정
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    print("✅ Testing Started! Press 'q' to quit, 'r' to recalibrate.")
    
    # FPS 계산용 변수
    prev_time = 0
    
    try:
        while True:
            # (1) 영상 캡처
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # (2) 얼굴 인식
            shape, face_rect = detector.detect(frame_bgr, gray)

            # (3) 상태 분석
            if analyzer.is_calibrating:
                analyzer.calibrate(frame_bgr, shape, face_rect)
                status = "CALIBRATING"
            else:
                status = analyzer.process(frame_bgr, shape, face_rect)

            # ============================================================
            # 📊 [TEST 전용] 디버깅 정보 화면 출력 (배경 없음)
            # ============================================================
            if shape is not None:
                # 1. 입 너비 (Mouth Width) 실시간 측정
                mouth_pts = shape[48:68]
                current_mouth_width = utils.get_mouth_width(mouth_pts)
                
                # 2. EAR (눈), MAR (입) 계산
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                ear = (utils.get_eye_aspect_ratio(left_eye) + utils.get_eye_aspect_ratio(right_eye)) / 2.0
                mar = analyzer.get_mouth_aspect_ratio(mouth_pts)

                # 3. 화면에 수치 그리기 (배경 박스 코드 삭제됨)
                
                # 가독성을 위해 글자색을 아주 선명한 색(하늘색, 라임색 등)으로 설정
                # [DEBUG MODE] 타이틀
                cv2.putText(frame_bgr, f"[DEBUG MODE]", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 상태 표시 (안전: 초록 / 위험: 빨강)
                status_color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
                cv2.putText(frame_bgr, f"Status: {status}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # 수치 데이터 (잘 보이게 흰색 진하게)
                cv2.putText(frame_bgr, f"EAR (Eye): {ear:.2f} (Th: {analyzer.EAR_THRESHOLD:.2f})", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"MAR (Mouth): {mar:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 입 너비 데이터 (웃음 감지 테스트용)
                # 평소보다 10% 이상 넓어지면 노란색으로 변함
                width_color = (255, 255, 255)
                if analyzer.normal_mouth_width > 0 and current_mouth_width > analyzer.normal_mouth_width * 1.1:
                    width_color = (0, 255, 255) # Yellow
                
                cv2.putText(frame_bgr, f"Width: {current_mouth_width:.1f} (Norm: {analyzer.normal_mouth_width:.1f})", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, width_color, 2)

            # FPS 표시 (우측 상단)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 화면 출력
            cv2.imshow("Test Mode (No Chatbot)", frame_bgr)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Recalibrating...")
                analyzer.is_calibrating = True
                analyzer.calib_ear_list = []
                # analyzer에 mouth 리스트가 있다면 초기화
                if hasattr(analyzer, 'calib_mouth_width_list'):
                    analyzer.calib_mouth_width_list = []

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()