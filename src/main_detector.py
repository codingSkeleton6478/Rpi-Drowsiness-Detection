import cv2
import dlib
import time
import numpy as np
from picamera2 import Picamera2
import utils 

print("Starting Class-Based Driver Monitor... 'q' to quit.")

class DriverMonitor:
    def __init__(self):
        # --- 1. 설정값 초기화 ---
        self.EAR_THRESHOLD = 0.3       # 학습 후 변경됨
        self.EAR_CONSEC_FRAMES = 40
        self.MAR_THRESHOLD = 0.5
        self.MAR_CONSEC_FRAMES = 15
        self.PITCH_THRESHOLD = 20.0
        self.NO_FACE_THRESHOLD = 50
        self.CALIBRATION_FRAMES = 100
        
        # --- 2. 상태 변수 ---
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.is_calibrating = True
        self.calib_ear_list = []
        
        # --- 3. 모델 로드 ---
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        # --- 4. 상수 및 좌표 ---
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # 자세 추정용 3D 모델
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype="double")
        
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def get_landmarks(self, gray, face):
        shape = self.predictor(gray, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    def calibrate(self, frame, faces, gray):
        """초기 5초간 운전자 눈 크기를 학습하는 함수"""
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Progress: {len(self.calib_ear_list)}/{self.CALIBRATION_FRAMES}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if len(faces) > 0:
            for face in faces:
                shape_np = self.get_landmarks(gray, face)
                
                left_ear = utils.get_eye_aspect_ratio(shape_np[self.LEFT_EYE])
                right_ear = utils.get_eye_aspect_ratio(shape_np[self.RIGHT_EYE])
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear > 0.15: # 눈 깜빡임 제외
                    self.calib_ear_list.append(avg_ear)
                
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 255), 2)

        # 학습 완료 체크
        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            # 상위 90% 지점 사용 (스마트 캘리브레이션)
            index_90th = int(len(self.calib_ear_list) * 0.90)
            normal_eye = self.calib_ear_list[index_90th]
            
            # 기준값 설정 (평소의 75%)
            calc_threshold = normal_eye * 0.75
            
            # [안전장치] 최소 0.21 보장
            self.EAR_THRESHOLD = max(0.21, calc_threshold)
            
            self.is_calibrating = False
            print(f"Calibration Done! Normal: {normal_eye:.3f}, Threshold: {self.EAR_THRESHOLD:.3f}")

    def monitor(self, frame, faces, gray):
        """실제 감시 모드"""
        if len(faces) == 0:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.counter_ear = 0
            self.counter_mar = 0
        else:
            self.counter_no_face = 0
            
            for face in faces:
                shape_np = self.get_landmarks(gray, face)

                # 1. 졸음 (EAR)
                left_pts = shape_np[self.LEFT_EYE]
                right_pts = shape_np[self.RIGHT_EYE]
                avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

                cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

                if avg_ear < self.EAR_THRESHOLD:
                    self.counter_ear += 1
                    if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.counter_ear = 0

                # 2. 하품 (MAR)
                # (utils에 MAR 함수가 없으면 여기서 직접 계산하거나 추가 필요. 여기선 직접 계산 예시)
                mouth = shape_np[self.MOUTH]
                A = np.linalg.norm(mouth[2] - mouth[10])
                B = np.linalg.norm(mouth[4] - mouth[8])
                C = np.linalg.norm(mouth[0] - mouth[6])
                mar = (A + B) / (2.0 * C)
                
                cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 255), 1)

                if mar > self.MAR_THRESHOLD:
                    self.counter_mar += 1
                    if self.counter_mar >= self.MAR_CONSEC_FRAMES:
                        cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    self.counter_mar = 0

                # 3. 자세 (Head Pose)
                img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
                (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                (rmat, _) = cv2.Rodrigues(rvec)
                proj_mat = np.hstack((rmat, tvec))
                euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
                pitch = euler[0][0]
                if pitch > 0: pitch -= 180

                # 화면 표시
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, f"Thresh: {self.EAR_THRESHOLD:.3f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if pitch > self.PITCH_THRESHOLD:
                    cv2.putText(frame, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def run(self):
        # 카메라 초기화
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        
        prev_time = 0
        
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            faces = self.detector(gray, 0)
            
            if self.is_calibrating:
                self.calibrate(frame_bgr, faces, gray)
            else:
                self.monitor(frame_bgr, faces, gray)
            
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Driver Monitor System", frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        picam2.stop()

# 실행 부분
if __name__ == "__main__":
    app = DriverMonitor()
    app.run()