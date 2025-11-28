import cv2
import numpy as np
import time
import src.utils as utils

class DriverAnalyzer:
    def __init__(self):
        # 1. 졸음 감지 (EAR)
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 60     # 기본 2초
        
        # 2. 하품 감지 (MAR)
        self.MAR_THRESHOLD = 0.5
        self.MAR_CONSEC_FRAMES = 30
        
        # 3. 고개 감지
        self.PITCH_THRESHOLD = 20.0
        self.YAW_THRESHOLD = 40.0
        self.NO_FACE_THRESHOLD = 50
        
        # [누락 방지] 캘리브레이션 프레임
        self.CALIBRATION_FRAMES = 50 

        # 4. 스마트 필터
        self.SMILE_RATIO = 1.3
        self.SMILE_COOLDOWN = 3.0
        self.CRITICAL_HOLD_TIME = 3.0

        # 5. 피로도 점수
        self.SCORE_WINDOW = 180
        self.FATIGUE_THRESHOLD = 80

        # [상태 변수]
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        
        self.normal_eye_size = 0.30
        self.normal_mouth_width = 0.0
        
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.sleep_trigger_count = 0  
        
        # [신규] Drowsiness Count (졸음 이력 관리)
        self.drowsiness_count = 0 
        
        # [신규] 대화 모드 플래그 (말하는 중인지 여부)
        self.is_speaking = False

        self.fatigue_events = []
        self.last_smile_time = 0
        self.last_critical_time = 0
        self.last_critical_status = "SAFE"

        # 모델 데이터
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
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
        # 대화 중(is_speaking)일 때는 점수 누적을 하지 않음 (억울한 누적 방지)
        if not self.is_speaking:
            self.fatigue_events.append((time.time(), score, reason))

    def get_current_fatigue_score(self):
        current_time = time.time()
        self.fatigue_events = [evt for evt in self.fatigue_events if current_time - evt[0] < self.SCORE_WINDOW]
        return sum(evt[1] for evt in self.fatigue_events)

    def refund_blink_points(self):
        current_time = time.time()
        self.fatigue_events = [
            evt for evt in self.fatigue_events
            if not ((current_time - evt[0] < 2.0) and (evt[2] == "Fast Blink"))
        ]

    # [Type A] 완전 초기화 (안 존다고 했을 때)
    def reset_full_calibration(self):
        print("🔄 [Analyzer] 완전 초기화 (점수 삭제 + 카운트 리셋)")
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        self.sleep_trigger_count = 0  
        self.counter_ear = 0
        self.fatigue_events = []
        self.drowsiness_count = 0  # 카운트도 0으로

    # [Type B] 부분 초기화 (졸음 인정했을 때)
    def reset_soft_for_next_stage(self):
        print("⚠️ [Analyzer] 부분 초기화 (점수는 비우지만, 감시 단계 격상)")
        # 캘리브레이션은 다시 하지 않음 (이미 졸린 상태라 눈 크기가 부정확할 수 있음)
        self.fatigue_events = [] # 점수는 비워서 챗봇 재발동 방지
        self.sleep_trigger_count = 0 
        self.counter_ear = 0
        self.drowsiness_count += 1 # 이력 추가 (재범 처리)

    def process(self, frame, shape_np, face_rect):
        current_status = "SAFE"
        
        if time.time() - self.last_critical_time < self.CRITICAL_HOLD_TIME:
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, f"!!! {self.last_critical_status} !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.last_critical_status

        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                return "NO_FACE"
            self.counter_ear = 0
            self.counter_mar = 0
            return "SAFE"
        self.counter_no_face = 0

        # Head Pose
        img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rmat, _) = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
        try: yaw = euler[1][0]
        except: yaw = 0
        
        is_head_turned = abs(yaw) > self.YAW_THRESHOLD
        if is_head_turned:
            cv2.putText(frame, f"SIDE LOOK ({int(yaw)})", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mouth & Smile
        mouth_pts = shape_np[self.MOUTH]
        current_mouth_width = utils.get_mouth_width(mouth_pts)
        is_smiling_now = False
        if self.normal_mouth_width > 0:
            if current_mouth_width > self.normal_mouth_width * self.SMILE_RATIO:
                is_smiling_now = True
                self.last_smile_time = time.time()
                self.refund_blink_points()

        is_smile_mode = is_smiling_now or (time.time() - self.last_smile_time < self.SMILE_COOLDOWN)
        if is_smile_mode:
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Yawn
        mouth_center = np.mean(mouth_pts, axis=0)
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.60] + mouth_center
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual.astype(np.int32))], -1, (0, 255, 255), 1)
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        is_yawning = False
        
        if mar > self.MAR_THRESHOLD and not is_smile_mode and not self.is_speaking:
            is_yawning = True 
            self.counter_mar += 1
            if self.counter_mar == self.MAR_CONSEC_FRAMES: 
                self.add_fatigue_point(30, "Yawn") 
                if current_status == "SAFE": current_status = "YAWN"
            elif self.counter_mar > self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.counter_mar = 0

        # EAR & Drowsiness
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        # =================================================================
        # [제안 A 핵심 구현] 대화 중(Speaking)일 때는 임계값을 2배로 늘림
        # 평소: 60프레임(2초) / 대화 중: 120프레임(4초)
        # =================================================================
        target_frames = self.EAR_CONSEC_FRAMES * 2 if self.is_speaking else self.EAR_CONSEC_FRAMES

        if avg_ear < self.EAR_THRESHOLD:
            if not is_smile_mode and not is_yawning and not is_head_turned:
                self.counter_ear += 1
                
                # 유동적인 임계값(target_frames) 사용
                if self.counter_ear >= target_frames:
                    
                    if self.sleep_trigger_count == 0:
                        cv2.putText(frame, "!!! 1st WARNING !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        if self.counter_ear == target_frames:
                            self.sleep_trigger_count += 1
                        current_status = "DROWSY" 
                        self.last_critical_time = time.time()
                        self.last_critical_status = "DROWSY"

                    else:
                        cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self.counter_ear == target_frames:
                            self.add_fatigue_point(60, "Long Blink")
                        current_status = "SLEEP"
                        self.last_critical_time = time.time()
                        self.last_critical_status = "SLEEP"

        else:
            if not is_smile_mode and not is_yawning and not is_head_turned:
                if self.counter_ear > 5 and self.counter_ear < target_frames:
                    # 대화 중이 아닐 때만 빠른 깜빡임 점수 누적
                    if not self.is_speaking:
                        self.add_fatigue_point(5, "Fast Blink") 
            self.counter_ear = 0

        # Score Evaluation
        total_score = self.get_current_fatigue_score()
        # 디버깅용: 현재 카운트 상태 표시
        info_text = f"Score: {total_score} | Count: {self.drowsiness_count} | Speak: {self.is_speaking}"
        cv2.putText(frame, info_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if current_status == "SAFE" or current_status == "YAWN":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        color = (0, 255, 0)
        if current_status == "SLEEP": color = (0, 0, 255)
        elif current_status in ["DROWSY", "YAWN"]: color = (0, 165, 255)
        
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), color, 2)
        
        return current_status

    def calibrate(self, frame, shape_np, face_rect):
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
            self.sleep_trigger_count = 0 
            print(f"✅ Calibration Done! Eye: {self.normal_eye_size:.3f}, Count: {self.drowsiness_count}")