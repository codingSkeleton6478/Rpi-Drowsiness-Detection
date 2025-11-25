import cv2
import dlib
import numpy as np
import src.utils as utils  # utils.py가 src 폴더 안에 있어야 함

class DriverMonitor:
    def __init__(self):
        # --- 설정값 ---
        self.EAR_THRESHOLD = 0.25      # 민감도 조절
        self.EAR_CONSEC_FRAMES = 20    # 몇 프레임 연속 감으면 졸음인가
        self.MAR_THRESHOLD = 0.5
        self.MAR_CONSEC_FRAMES = 15
        self.PITCH_THRESHOLD = 20.0
        self.NO_FACE_THRESHOLD = 50
        self.CALIBRATION_FRAMES = 100
        
        # --- 상태 변수 ---
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.is_calibrating = True
        self.calib_ear_list = []
        
        # --- 모델 로드 ---
        # 실행 위치(main.py) 기준 경로 주의
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        # --- 상수 ---
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Head Pose용 3D 모델 점
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
        """초기 캘리브레이션"""
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        if len(faces) > 0:
            for face in faces:
                shape_np = self.get_landmarks(gray, face)
                left_ear = utils.get_eye_aspect_ratio(shape_np[self.LEFT_EYE])
                right_ear = utils.get_eye_aspect_ratio(shape_np[self.RIGHT_EYE])
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear > 0.15:
                    self.calib_ear_list.append(avg_ear)
                
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 255), 2)

        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            index_90th = int(len(self.calib_ear_list) * 0.90)
            normal_eye = self.calib_ear_list[index_90th]
            self.EAR_THRESHOLD = max(0.21, normal_eye * 0.75)
            self.is_calibrating = False
            print(f"Calibration Done! Thresh: {self.EAR_THRESHOLD:.3f}")

    def process_frame(self, frame, faces, gray):
        """
        프레임 한 장을 처리하고 현재 상태(status)를 반환함
        Returns: "SAFE", "SLEEP", "YAWN", "HEAD_DOWN", "NO_FACE"
        """
        status = "SAFE"

        if len(faces) == 0:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return "NO_FACE"
            self.counter_ear = 0
            return "SAFE"
        
        self.counter_no_face = 0
        
        for face in faces:
            shape_np = self.get_landmarks(gray, face)

            # 1. EAR (졸음)
            left_pts = shape_np[self.LEFT_EYE]
            right_pts = shape_np[self.RIGHT_EYE]
            avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

            # 눈 그리기
            cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

            if avg_ear < self.EAR_THRESHOLD:
                self.counter_ear += 1
                if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    status = "SLEEP"
            else:
                self.counter_ear = 0

            # 2. MAR (하품) - utils에 함수가 없다면 직접 계산
            mouth = shape_np[self.MOUTH]
            A = np.linalg.norm(mouth[2] - mouth[10])
            B = np.linalg.norm(mouth[4] - mouth[8])
            C = np.linalg.norm(mouth[0] - mouth[6])
            mar = (A + B) / (2.0 * C)

            if mar > self.MAR_THRESHOLD:
                self.counter_mar += 1
                if self.counter_mar >= self.MAR_CONSEC_FRAMES:
                    cv2.putText(frame, "!!! YAWN !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if status != "SLEEP": status = "YAWN"
            else:
                self.counter_mar = 0

            # 3. Head Pose (고개 숙임)
            img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
            (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            (rmat, _) = cv2.Rodrigues(rvec)
            proj_mat = np.hstack((rmat, tvec))
            euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
            pitch = euler[0][0]
            if pitch > 0: pitch -= 180

            if pitch > self.PITCH_THRESHOLD:
                 cv2.putText(frame, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 if status == "SAFE": status = "HEAD_DOWN"

            # 정보 표시
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        return status