import cv2
import numpy as np
import src.utils as utils

class DriverAnalyzer:
    def __init__(self):
        # 설정값 (원본 유지)
        self.EAR_THRESHOLD = 0.25      
        self.EAR_CONSEC_FRAMES = 15    
        self.MAR_THRESHOLD = 0.5       
        self.MAR_CONSEC_FRAMES = 10     
        self.PITCH_THRESHOLD = 20.0    
        self.YAW_THRESHOLD = 30.0      
        self.NO_FACE_THRESHOLD = 50    
        
        self.CALIBRATION_FRAMES = 50   
        self.is_calibrating = True
        self.calib_ear_list = []
        
        self.normal_eye_size = 0.30 
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.last_mar_val = 0.0 

        # 인덱스
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Head Pose 3D 모델
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype="double")
        
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def get_mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def process(self, frame, shape_np, face_rect):
        """
        Detector가 준 shape_np(배열)를 바로 사용합니다.
        """
        status = "SAFE"

        # 1. 얼굴 없음 처리
        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return "NO_FACE"
            self.counter_ear = 0
            self.counter_mar = 0
            return "SAFE"
        
        self.counter_no_face = 0
        
        # [수정됨] shape_np는 이미 Numpy 배열이므로 변환 함수 호출 제거함!
        
        # 1. 졸음 (EAR)
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        if avg_ear < self.EAR_THRESHOLD:
            self.counter_ear += 1
            if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                status = "SLEEP"
        else:
            self.counter_ear = 0

        # 2. 하품 (MAR)
        mouth_pts = shape_np[self.MOUTH]
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        
        mouth_center = np.mean(mouth_pts, axis=0)
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.80] + mouth_center
        mouth_pts_visual = mouth_pts_visual.astype(np.int32)
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual)], -1, (0, 255, 255), 1)

        if mar > self.MAR_THRESHOLD:
            self.counter_mar += 1
            if self.counter_mar >= self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if status != "SLEEP": status = "YAWN"
        else:
            self.counter_mar = 0

        # 3. 자세 (Head Pose)
        img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rmat, _) = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
        pitch = euler[0][0]
        try: yaw = euler[1][0] 
        except: yaw = 0

        if pitch > 0: pitch -= 180

        if abs(yaw) > self.YAW_THRESHOLD:
            cv2.putText(frame, "LOOKING AWAY", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            self.counter_ear = 0
        
        if pitch > self.PITCH_THRESHOLD:
             cv2.putText(frame, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             if status == "SAFE": status = "HEAD_DOWN"

        # 박스 그리기
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f} / Thresh: {self.EAR_THRESHOLD:.3f}", (face_rect.left(), face_rect.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        return status

    def calibrate(self, frame, shape_np, face_rect):
        """캘리브레이션"""
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Progress: {len(self.calib_ear_list)}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if shape_np is not None:
            # 여기도 shape_np를 바로 씁니다
            left_pts = shape_np[self.LEFT_EYE]
            right_pts = shape_np[self.RIGHT_EYE]
            avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0
            
            if avg_ear > 0.15:
                self.calib_ear_list.append(avg_ear)
            
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 255), 2)

        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            index_90th = int(len(self.calib_ear_list) * 0.90)
            self.normal_eye_size = self.calib_ear_list[index_90th]
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)
            self.is_calibrating = False
            print(f"✅ Calibration Done! Normal: {self.normal_eye_size:.3f}, Threshold: {self.EAR_THRESHOLD:.3f}")