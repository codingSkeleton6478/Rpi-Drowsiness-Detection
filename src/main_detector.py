import cv2
import dlib
import time
import numpy as np
import math
from picamera2 import Picamera2
import utils 

print("Starting Final Tuned Monitor... 'q' to quit.")

# =================================================================
# 1. 설정값 (초기값)
# =================================================================
EAR_THRESHOLD = 0.3       # (학습 후 자동 변경됨)
EAR_CONSEC_FRAMES = 40    # 졸음 지속 프레임 (약 2초)

MAR_THRESHOLD = 0.5       # 하품 기준 (입 벌림)
MAR_CONSEC_FRAMES = 15    

PITCH_THRESHOLD = 20.0    # 고개 숙임 기준 (도)
NO_FACE_THRESHOLD = 50    # 얼굴 실종 기준 (약 2.5초)

# 학습 설정
CALIBRATION_FRAMES = 100  # 학습할 프레임 수 (약 3~5초)
IS_CALIBRATING = True     
calib_ear_list = []       

# 카운터
COUNTER_EAR = 0     
COUNTER_MAR = 0    
COUNTER_NO_FACE = 0 

# 2. 모델 로드
landmark_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# 3. 3D 모델 좌표 (자세 추정용)
model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype="double")

focal_length = 640
center = (320, 240)
camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4,1)) 

def shape_to_np(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

def get_mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10]) 
    B = np.linalg.norm(mouth[4] - mouth[8])  
    C = np.linalg.norm(mouth[0] - mouth[6])  
    return (A + B) / (2.0 * C)

# 4. 카메라 시작
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

prev_time = 0

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    # -------------------------------------------------------------
    # [MODE 1] 캘리브레이션 (초기 학습)
    # -------------------------------------------------------------
    if IS_CALIBRATING:
        cv2.putText(frame_bgr, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame_bgr, f"Progress: {len(calib_ear_list)}/{CALIBRATION_FRAMES}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if len(faces) > 0:
            for face in faces:
                shape = predictor(gray, face)
                shape_np = shape_to_np(shape)
                
                left_pts = shape_np[LEFT_EYE]
                right_pts = shape_np[RIGHT_EYE]
                left_ear = utils.get_eye_aspect_ratio(left_pts)
                right_ear = utils.get_eye_aspect_ratio(right_pts)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # 이상치(눈 깜빡임 0.15 이하) 제외하고 수집
                if avg_ear > 0.15: 
                    calib_ear_list.append(avg_ear)
                
                cv2.rectangle(frame_bgr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 255), 2)

        # 학습 종료 조건 달성
        if len(calib_ear_list) >= CALIBRATION_FRAMES:
            # [스마트 알고리즘] 상위 90% 값 사용 (가장 잘 떴을 때 기준)
            calib_ear_list.sort()
            index_90th = int(len(calib_ear_list) * 0.90)
            normal_eye_openness = calib_ear_list[index_90th]
            
            # 기준값 설정 (평소 크기의 75%)
            # 예: 평소 0.31 -> 기준 0.23 (여유 있음)
            calculated_threshold = normal_eye_openness * 0.75
            
            # [안전장치 완화] 최소 0.21 보장 (0.25는 너무 높았음)
            # 사용자님 눈이 감았을 때 0.21 정도 되므로, 기준은 0.21 이상이어야 함.
            if calculated_threshold < 0.21:
                EAR_THRESHOLD = 0.21
            else:
                EAR_THRESHOLD = calculated_threshold
            
            IS_CALIBRATING = False
            print(f"Calibration Done! Max Open: {normal_eye_openness:.3f}, Threshold: {EAR_THRESHOLD:.3f}")
    
    # -------------------------------------------------------------
    # [MODE 2] 실시간 감시
    # -------------------------------------------------------------
    else:
        if len(faces) == 0:
            # 얼굴 실종 (전방 주시 태만)
            COUNTER_NO_FACE += 1
            if COUNTER_NO_FACE >= NO_FACE_THRESHOLD:
                cv2.putText(frame_bgr, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            COUNTER_EAR = 0
            COUNTER_MAR = 0
        else:
            COUNTER_NO_FACE = 0

            for face in faces:
                shape = predictor(gray, face)
                shape_np = shape_to_np(shape)

                # A. 졸음 (EAR)
                left_pts = shape_np[LEFT_EYE]
                right_pts = shape_np[RIGHT_EYE]
                left_ear = utils.get_eye_aspect_ratio(left_pts)
                right_ear = utils.get_eye_aspect_ratio(right_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                cv2.drawContours(frame_bgr, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame_bgr, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

                if avg_ear < EAR_THRESHOLD:
                    COUNTER_EAR += 1
                    if COUNTER_EAR >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame_bgr, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER_EAR = 0

                # B. 하품 (MAR)
                mouth_pts = shape_np[MOUTH]
                mar = get_mouth_aspect_ratio(mouth_pts)
                cv2.drawContours(frame_bgr, [cv2.convexHull(mouth_pts)], -1, (0, 255, 255), 1)

                if mar > MAR_THRESHOLD:
                    COUNTER_MAR += 1
                    if COUNTER_MAR >= MAR_CONSEC_FRAMES:
                        cv2.putText(frame_bgr, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    COUNTER_MAR = 0

                # C. 자세 (Head Pose)
                img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
                (success, rvec, tvec) = cv2.solvePnP(model_points, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (rmat, _) = cv2.Rodrigues(rvec)
                proj_mat = np.hstack((rmat, tvec))
                euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
                pitch = euler[0][0]
                if pitch > 0: pitch -= 180 

                cv2.rectangle(frame_bgr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                
                # 정보 표시 (학습된 임계값도 보여줌)
                cv2.putText(frame_bgr, f"Thresh: {EAR_THRESHOLD:.3f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame_bgr, f"EAR: {avg_ear:.3f}", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if pitch > PITCH_THRESHOLD:
                     cv2.putText(frame_bgr, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Smart Driver Monitor", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
print("Detector 종료.")