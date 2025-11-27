import cv2
import numpy as np
import time
import src.utils as utils

class DriverAnalyzer:
    def __init__(self):
        # ==========================================
        # 1. 임계값 및 설정 (Settings)
        # ==========================================
        self.EAR_THRESHOLD = 0.25       # 눈 감김 기준 (개인화됨)
        
        # 60프레임 = 약 2초 동안 눈을 감아야 'SLEEP'으로 판정
        self.EAR_CONSEC_FRAMES = 60     
        
        self.MAR_THRESHOLD = 0.5        # 하품 기준
        self.MAR_CONSEC_FRAMES = 30     # 하품 지속 시간 (약 1초)
        
        self.PITCH_THRESHOLD = 20.0     # 고개 숙임 각도 (사용은 안 하지만 변수는 유지)
        self.YAW_THRESHOLD = 40.0       # 고개 돌림 각도
        self.NO_FACE_THRESHOLD = 50     # 얼굴 사라짐 허용 프레임
        
        # 캘리브레이션 변수
        self.CALIBRATION_FRAMES = 50   
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = [] 
        
        self.normal_eye_size = 0.30 
        self.normal_mouth_width = 0.0    
        
        # 카운터 변수
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        
        # ==========================================
        # 2. 피로도 점수 시스템
        # ==========================================
        self.fatigue_events = []         
        self.SCORE_WINDOW = 180          # 3분 기억
        self.FATIGUE_THRESHOLD = 80      # 챗봇 호출 기준 점수
        
        # ==========================================
        # 3. 스마트 감지 필터 (Smile & Latching)
        # ==========================================
        self.SMILE_RATIO = 1.3           
        self.last_smile_time = 0         
        self.SMILE_COOLDOWN = 3.0        # 웃음 잔상 3초
        
        # 위험 상태 유지(Latching) 타이머
        self.last_critical_time = 0      
        self.CRITICAL_HOLD_TIME = 3.0    # 3초간 위험 상태 유지
        self.last_critical_status = "SAFE" 
        
        # 랜드마크 인덱스
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Head Pose 3D 모델 점
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

    def add_fatigue_point(self, score, reason):
        # [수정] 환불을 위해 reason(사유)도 함께 저장
        self.fatigue_events.append((time.time(), score, reason))

    def get_current_fatigue_score(self):
        current_time = time.time()
        # 3분 지난 기록 삭제
        self.fatigue_events = [evt for evt in self.fatigue_events if current_time - evt[0] < self.SCORE_WINDOW]
        return sum(evt[1] for evt in self.fatigue_events)

    def refund_blink_points(self):
        """[기능] 웃음 감지 시 최근 2초간의 'Fast Blink' 점수 삭제"""
        current_time = time.time()
        self.fatigue_events = [
            evt for evt in self.fatigue_events
            if not ((current_time - evt[0] < 2.0) and (evt[2] == "Fast Blink"))
        ]

    def process(self, frame, shape_np, face_rect):
        current_status = "SAFE"
        
        # ==========================================
        # [Step 0] 위험 상태 유지 (Latching) 체크
        # ==========================================
        if time.time() - self.last_critical_time < self.CRITICAL_HOLD_TIME:
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, f"!!! {self.last_critical_status} !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.last_critical_status

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

        # ==========================================
        # [Step 1] 각도 계산 (맨 위로 이동됨)
        # ==========================================
        img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rmat, _) = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
        
        # [신규] Yaw(좌우) 값 추출
        try: yaw = euler[1][0]
        except: yaw = 0
        
        # Yaw Filter 적용: 고개 30도 이상 돌림
        is_head_turned = abs(yaw) > self.YAW_THRESHOLD  

        if is_head_turned:
            cv2.putText(frame, f"SIDE LOOK ({int(yaw)})", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ==========================================
        # [Step 2] 스마일 감지 (Smile Filter)
        # ==========================================
        mouth_pts = shape_np[self.MOUTH]
        current_mouth_width = utils.get_mouth_width(mouth_pts)
        
        is_smiling_now = False
        if self.normal_mouth_width > 0:
            if current_mouth_width > self.normal_mouth_width * self.SMILE_RATIO:
                is_smiling_now = True
                self.last_smile_time = time.time()
                self.refund_blink_points() # 환불 실행

        # 웃음 잔상(쿨다운) 적용
        is_smile_mode = is_smiling_now or (time.time() - self.last_smile_time < self.SMILE_COOLDOWN)

        if is_smile_mode:
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ==========================================
        # [Step 3] 하품 감지 (Yawn)
        # ==========================================
        mouth_center = np.mean(mouth_pts, axis=0)
        # [수정] 입술 높이 비율 0.60으로 축소
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.60] + mouth_center
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual.astype(np.int32))], -1, (0, 255, 255), 1)
        
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        is_yawning = False
        
        if mar > self.MAR_THRESHOLD and not is_smile_mode:
            is_yawning = True 
            self.counter_mar += 1
            if self.counter_mar == self.MAR_CONSEC_FRAMES: 
                self.add_fatigue_point(30, "Yawn") # 점수 30점
                if current_status == "SAFE": current_status = "YAWN"
            elif self.counter_mar > self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.counter_mar = 0

        # ==========================================
        # [Step 4] 졸음 감지 (EAR / Sleep)
        # ==========================================
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        # 적응형 임계값 업데이트
        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        if avg_ear < self.EAR_THRESHOLD:
            # [필터 적용] 웃거나, 하품하거나, 고개 돌릴 땐 점수 안 줌
            if not is_smile_mode and not is_yawning and not is_head_turned:
                self.counter_ear += 1
                
                if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if self.counter_ear == self.EAR_CONSEC_FRAMES:
                        self.add_fatigue_point(60, "Long Blink") # 점수 60점
                    
                    current_status = "SLEEP"
                    self.last_critical_time = time.time()
                    self.last_critical_status = "SLEEP"
        else:
            # 눈 떴을 때
            if not is_smile_mode and not is_yawning and not is_head_turned:
                if self.counter_ear > 5 and self.counter_ear < self.EAR_CONSEC_FRAMES:
                    self.add_fatigue_point(5, "Fast Blink") # 점수 5점
            self.counter_ear = 0


        # ==========================================
        # [Step 6] 점수 기반 경고 (DROWSY)
        # ==========================================
        total_score = self.get_current_fatigue_score()
        cv2.putText(frame, f"Fatigue Score: {total_score}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_status == "SAFE" or current_status == "YAWN":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        # 화면 박스 그리기
        color = (0, 255, 0) # Green
        if current_status == "SLEEP": color = (0, 0, 255) # Red
        elif current_status in ["DROWSY", "YAWN"]: color = (0, 165, 255) # Orange
        
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), color, 2)
        
        return current_status

    def calibrate(self, frame, shape_np, face_rect):
        """캘리브레이션 (눈 + 입 너비)"""
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Progress: {len(self.calib_ear_list)}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if shape_np is not None:
            left_pts = shape_np[self.LEFT_EYE]
            right_pts = shape_np[self.RIGHT_EYE]
            avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0
            if avg_ear > 0.15: 
                self.calib_ear_list.append(avg_ear)
            
            mouth_pts = shape_np[self.MOUTH]
            mouth_width = utils.get_mouth_width(mouth_pts)
            self.calib_mouth_width_list.append(mouth_width)

            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 255), 2)

        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            index_90th = int(len(self.calib_ear_list) * 0.90)
            self.normal_eye_size = self.calib_ear_list[index_90th]
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

            if self.calib_mouth_width_list:
                self.calib_mouth_width_list.sort()
                mid_index = len(self.calib_mouth_width_list) // 2
                self.normal_mouth_width = self.calib_mouth_width_list[mid_index]
            
            self.is_calibrating = False
            print(f"✅ Calibration Done! Eye Thresh: {self.EAR_THRESHOLD:.3f}, Mouth Norm: {self.normal_mouth_width:.1f}")