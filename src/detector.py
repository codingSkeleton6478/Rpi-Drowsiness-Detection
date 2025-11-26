import cv2
import dlib
import numpy as np
import src.utils as utils

class DriverMonitor:
    def __init__(self):
        # ==========================================
        # 1. 설정값
        # ==========================================
        self.EAR_THRESHOLD = 0.25      
        self.EAR_CONSEC_FRAMES = 15    
        
        self.MAR_THRESHOLD = 0.5       
        self.MAR_CONSEC_FRAMES = 10     
        
        self.PITCH_THRESHOLD = 20.0    
        self.YAW_THRESHOLD = 30.0      
        self.NO_FACE_THRESHOLD = 50    
        
        # 캘리브레이션
        self.CALIBRATION_FRAMES = 50   
        self.is_calibrating = True
        self.calib_ear_list = []
        
        # [NEW] 적응형 알고리즘 변수
        self.normal_eye_size = 0.30 # 초기값 (나중에 캘리브레이션으로 덮어씌워짐)
        
        # 카운터
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        
        self.last_mar_val = 0.0 

        # ==========================================
        # 2. 모델 로드 (Dlib 순정)
        # ==========================================
        print("✅ Loading Dlib Detector (Smart Adaptive Mode)...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype="double")
        
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def expand_rect(self, rect, frame_shape):
        img_h, img_w = frame_shape[:2]
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        pad_top = int(h * 0.05)
        pad_bottom = int(h * 0.25) 
        pad_side = int(w * 0.15)
        
        new_x1 = max(0, x - pad_side)
        new_y1 = max(0, y - pad_top)
        new_x2 = min(img_w, x + w + pad_side)
        new_y2 = min(img_h, y + h + pad_bottom)
        
        return dlib.rectangle(new_x1, new_y1, new_x2, new_y2)

    def get_landmarks(self, gray, face):
        shape = self.predictor(gray, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    def get_mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def calibrate(self, frame, gray):
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Progress: {len(self.calib_ear_list)}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        faces = self.detector(gray, 0)

        if len(faces) > 0:
            for face in faces:
                expanded_face = self.expand_rect(face, frame.shape)
                try:
                    shape_np = self.get_landmarks(gray, expanded_face)
                    left_pts = shape_np[self.LEFT_EYE]
                    right_pts = shape_np[self.RIGHT_EYE]
                    avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0
                    
                    if avg_ear > 0.15:
                        self.calib_ear_list.append(avg_ear)
                    
                    cv2.rectangle(frame, (expanded_face.left(), expanded_face.top()), (expanded_face.right(), expanded_face.bottom()), (0, 255, 255), 2)
                except:
                    continue

        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            index_90th = int(len(self.calib_ear_list) * 0.90)
            
            # [NEW] 평소 눈 크기를 저장해둠
            self.normal_eye_size = self.calib_ear_list[index_90th]
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)
            
            self.is_calibrating = False
            print(f"✅ Calibration Done! Normal: {self.normal_eye_size:.3f}, Threshold: {self.EAR_THRESHOLD:.3f}")

    def process_frame(self, frame, gray):
        status = "SAFE"
        faces = self.detector(gray, 0)

        if len(faces) == 0:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return "NO_FACE"
            self.counter_ear = 0
            self.counter_mar = 0
        else:
            self.counter_no_face = 0
            
            for face in faces:
                expanded_face = self.expand_rect(face, frame.shape)
                
                try:
                    shape_np = self.get_landmarks(gray, expanded_face)
                except:
                    continue

                # 1. 졸음
                left_pts = shape_np[self.LEFT_EYE]
                right_pts = shape_np[self.RIGHT_EYE]
                avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

                # -----------------------------------------------------------
                # [핵심 기능] 적응형 업데이트 (Smart Adaptive Update)
                # 눈을 '확실히' 뜨고 있다면(기준값보다 큼), 평소 눈 크기 정보를 아주 천천히 업데이트함
                # -----------------------------------------------------------
                if avg_ear > self.EAR_THRESHOLD:
                    # 현재 EAR을 1% 비중으로 반영하여 평소 눈 크기 갱신
                    self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
                    # 기준값도 같이 갱신 (평소 눈 크기의 85% 유지)
                    self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)
                # -----------------------------------------------------------

                cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

                if avg_ear < self.EAR_THRESHOLD:
                    self.counter_ear += 1
                    if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        status = "SLEEP"
                else:
                    self.counter_ear = 0

                # 2. 하품
                mouth_pts = shape_np[self.MOUTH]
                mar = self.get_mouth_aspect_ratio(mouth_pts)
                self.last_mar_val = mar
                
                # 입술 축소 그리기
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

                # 3. 자세
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
                    continue

                if pitch > self.PITCH_THRESHOLD:
                     cv2.putText(frame, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     if status == "SAFE": status = "HEAD_DOWN"

                # 박스 그리기
                cv2.rectangle(frame, (expanded_face.left(), expanded_face.top()), (expanded_face.right(), expanded_face.bottom()), (0, 255, 0), 2)
                # [정보 표시] 변하는 기준값(Thresh)을 보여줍니다
                cv2.putText(frame, f"EAR: {avg_ear:.2f} / Thresh: {self.EAR_THRESHOLD:.3f}", (expanded_face.left(), expanded_face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        return status