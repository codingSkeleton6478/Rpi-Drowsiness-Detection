import cv2
import numpy as np
import time
import src.utils as utils

class DriverAnalyzer:
    def __init__(self):
        # ==========================================
        # 1. 임계값 및 설정 (Settings)
        # ==========================================
        self.EAR_THRESHOLD = 0.25       # 눈 감김 기준
        
        # [수정] 30프레임 (약 1초) 동안 눈 감아야 SLEEP 판정
        self.EAR_CONSEC_FRAMES = 30     
        
        self.MAR_THRESHOLD = 0.5        # 하품 기준
        # [수정] 30프레임 (약 1초) 유지해야 하품 인정 (오작동 방지)
        self.MAR_CONSEC_FRAMES = 30     
        
        self.PITCH_THRESHOLD = 20.0     # 고개 숙임
        self.YAW_THRESHOLD = 30.0       # 고개 돌림
        self.NO_FACE_THRESHOLD = 50     
        
        # 캘리브레이션 변수
        self.CALIBRATION_FRAMES = 50   
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = [] 
        
        self.normal_eye_size = 0.30 
        self.normal_mouth_width = 0.0    
        
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        
        # ==========================================
        # 2. 피로도 점수 밸런스 패치 (Re-balancing)
        # ==========================================
        self.fatigue_events = []         
        self.SCORE_WINDOW = 180          # 3분간 기록 유지
        
        # [상향] 챗봇이 말을 거는 기준 점수 (60 -> 80)
        self.FATIGUE_THRESHOLD = 80      
        
        self.SMILE_RATIO = 1.2           
        self.last_smile_time = 0         
        
        # [수정] 웃음 쿨다운 3초 (10초는 너무 길어서 위험)
        self.SMILE_COOLDOWN = 3.0        
        
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
        self.fatigue_events.append((time.time(), score))
        # 디버깅용: 실제 환경에선 주석 처리해도 됨
        # print(f"⚠️ [Fatigue] {reason} (+{score})") 

    def get_current_fatigue_score(self):
        current_time = time.time()
        # 3분 지난 기록 삭제
        self.fatigue_events = [evt for evt in self.fatigue_events if current_time - evt[0] < self.SCORE_WINDOW]
        total_score = sum(evt[1] for evt in self.fatigue_events)
        return total_score

    def process(self, frame, shape_np, face_rect):
        # 리턴할 상태값 (기본: SAFE)
        # 우선순위: HEAD_DOWN > SLEEP > DROWSY > YAWN > SAFE
        current_status = "SAFE"
        
        # 1. 얼굴 없음
        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return "NO_FACE"
            self.counter_ear = 0
            self.counter_mar = 0
            return "SAFE"
        self.counter_no_face = 0

        # 데이터 추출
        mouth_pts = shape_np[self.MOUTH]
        current_mouth_width = utils.get_mouth_width(mouth_pts)
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        
        # 2. 스마일 감지
        is_smiling_now = False
        if self.normal_mouth_width > 0:
            if current_mouth_width > self.normal_mouth_width * self.SMILE_RATIO:
                is_smiling_now = True
                self.last_smile_time = time.time() 

        # 쿨다운(잔상) 포함 웃음 모드 체크
        is_smile_mode = is_smiling_now or (time.time() - self.last_smile_time < self.SMILE_COOLDOWN)

        if is_smile_mode:
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. 하품 감지
        mouth_center = np.mean(mouth_pts, axis=0)
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.80] + mouth_center
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual.astype(np.int32))], -1, (0, 255, 255), 1)
        
        is_yawning = False
        if mar > self.MAR_THRESHOLD and not is_smile_mode:
            is_yawning = True 
            self.counter_mar += 1
            if self.counter_mar == self.MAR_CONSEC_FRAMES: 
                # [하향] 하품 점수 40 -> 30점
                self.add_fatigue_point(30, "Yawn") 
                # 하품은 상태만 리턴하고, 챗봇 호출은 점수 기반으로 함
                if current_status == "SAFE": current_status = "YAWN"
            elif self.counter_mar > self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.counter_mar = 0

        # 4. 졸음 감지 (EAR)
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        if avg_ear < self.EAR_THRESHOLD:
            # 웃음/하품 중이 아닐 때만 체크
            if not is_smile_mode and not is_yawning:
                self.counter_ear += 1
                if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                    # 1초 이상 눈 감음 -> 즉시 위험 (SLEEP)
                    cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.counter_ear == self.EAR_CONSEC_FRAMES:
                        # [유지] 수면은 60점 (한방에 위험 수준 근접)
                        self.add_fatigue_point(60, "Long Blink")
                    current_status = "SLEEP" # 최우선 순위
        else:
            # 눈 떴을 때 빈도 체크
            if not is_smile_mode and not is_yawning:
                if self.counter_ear > 5 and self.counter_ear < self.EAR_CONSEC_FRAMES:
                    # [대폭 하향] 단순 깜빡임 15 -> 5점
                    self.add_fatigue_point(5, "Fast Blink") 
            self.counter_ear = 0

        # 5. 고개 숙임 (Head Pose)
        img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rmat, _) = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
        pitch = euler[0][0]
        if pitch > 0: pitch -= 180

        if pitch > self.PITCH_THRESHOLD:
             cv2.putText(frame, "!!! HEAD DOWN !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             current_status = "HEAD_DOWN" # 최우선 순위

        # 6. 최종 점수 계산 및 DROWSY 판정
        total_score = self.get_current_fatigue_score()
        cv2.putText(frame, f"Fatigue Score: {total_score}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # SLEEP이나 HEAD_DOWN이 아닐 때만 점수 기반 경고(DROWSY) 내보냄
        if current_status == "SAFE" or current_status == "YAWN":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)
        
        return current_status
    
    # calibrate 함수는 변경 없음 (생략 가능하나 전체 덮어쓰기 위해 유지)
    def calibrate(self, frame, shape_np, face_rect):
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