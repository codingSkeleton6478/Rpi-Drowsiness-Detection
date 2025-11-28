import cv2
import numpy as np
import time
import src.utils as utils

class DriverAnalyzer:
    """
    운전자 상태 모니터링 및 졸음/부주의 감지 클래스
    - 기능: 졸음(EAR), 하품(MAR), 고개 돌림(Head Pose), 웃음 감지 필터링
    - 특징: 캘리브레이션을 통한 개인화 임계값 적용
    """

    def __init__(self):
        # ==========================================
        # [Configuration] 임계값 및 설정 상수
        # ==========================================
        # 1. 졸음 감지 (EAR)
        self.EAR_THRESHOLD = 0.25       # 눈 감김 기준 (캘리브레이션 후 조정됨)
        self.EAR_CONSEC_FRAMES = 60     # 약 2초 (60프레임) 지속 시 졸음 판정
        
        # 2. 하품 감지 (MAR)
        self.MAR_THRESHOLD = 0.6        # [수정] 0.5 -> 0.6 (하품 오탐 방지를 위해 상향)
        self.MAR_CONSEC_FRAMES = 30     # 약 1초 지속 시 하품 판정
        
        # 3. 고개 감지 (Head Pose)
        self.PITCH_THRESHOLD = 20.0     # 고개 숙임 (Sleeping)
        self.YAW_THRESHOLD = 40.0       # 고개 돌림 (Distraction)
        self.NO_FACE_THRESHOLD = 50     # 얼굴 소실 허용 프레임 수
        self.CALIBRATION_FRAMES = 50    # 캘리브레이션에 필요한 프레임 수

        # 4. 스마트 필터 (웃음 & 상태 유지)
        self.SMILE_RATIO = 1.3          # 평소 입 너비 대비 1.3배면 웃음으로 간주
        self.SMILE_COOLDOWN = 3.0       # 웃음 감지 후 3초간 졸음 판단 중지
        self.CRITICAL_HOLD_TIME = 3.0   # 위험 상태(졸음 등) 판정 후 3초간 경고 유지 (Latching)

        # 5. 피로도 점수 시스템
        self.SCORE_WINDOW = 180         # 최근 3분(180초) 데이터만 계산
        self.FATIGUE_THRESHOLD = 80     # 이 점수를 넘으면 'DROWSY' 확정

        # ==========================================
        # [State] 상태 관리 변수
        # ==========================================
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        
        self.normal_eye_size = 0.30     # 평소 눈 크기 (업데이트됨)
        self.normal_mouth_width = 0.0   # 평소 입 너비 (캘리브레이션됨)
        
        # 프레임 카운터
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.sleep_trigger_count = 0    # 1차 경고(0) -> 2차 비상(1) 상태 추적

        self.fatigue_events = []        # 피로도 누적 리스트 [(time, score, reason), ...]
        
        # [신규] 핵심 요구사항 변수 추가
        self.drowsiness_count = 0       # 졸음 재범 횟수 관리 (Track B)
        self.is_speaking = False        # 챗봇 대화 중 여부 (동적 임계값용)

        self.last_smile_time = 0
        self.last_critical_time = 0
        self.last_critical_status = "SAFE"

        # ==========================================
        # [Camera & Model] 모델 상수
        # ==========================================
        # dlib 68 랜드마크 인덱스
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # PnP Solver용 3D 모델 포인트 (Generic Face Model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype="double")
        
        # 카메라 매트릭스 (RPi Camera V3 Wide 추정치)
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def get_mouth_aspect_ratio(self, mouth):
        """입의 종횡비(MAR) 계산: 높이 / 너비"""
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def add_fatigue_point(self, score, reason):
        """피로도 점수 추가 및 로그 기록"""
        # [수정] 대화 중(is_speaking)일 때는 억울한 점수 누적 방지
        if not self.is_speaking:
            self.fatigue_events.append((time.time(), score, reason))

    def get_current_fatigue_score(self):
        """최근 SCORE_WINDOW 시간 내의 피로도 총합 계산"""
        current_time = time.time()
        # 유효 기간 지난 점수 제거 (Sliding Window)
        self.fatigue_events = [evt for evt in self.fatigue_events if current_time - evt[0] < self.SCORE_WINDOW]
        return sum(evt[1] for evt in self.fatigue_events)

    def refund_blink_points(self):
        """오인식 방지: 웃음 감지 시 최근 2초간의 'Fast Blink' 점수 무효화"""
        current_time = time.time()
        self.fatigue_events = [
            evt for evt in self.fatigue_events
            if not ((current_time - evt[0] < 2.0) and (evt[2] == "Fast Blink"))
        ]

    # [수정] 함수명 변경 및 졸음 이력 초기화 로직 추가
    def reset_full_calibration(self):
        """사용자 변경 또는 '부정(DENY)' 시 완전 초기화"""
        print("🔄 [Analyzer] System Full Reset (Score + Count).")
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        self.sleep_trigger_count = 0  
        self.counter_ear = 0
        self.fatigue_events = []
        self.drowsiness_count = 0  # [신규] 재범 카운트도 초기화

    # [신규] 졸음 인정 시 호출되는 부분 초기화
    def reset_soft_for_next_stage(self):
        """졸음 '인정(ADMIT)' 시 점수는 비우고 감시 단계 격상"""
        print("⚠️ [Analyzer] Soft Reset (Count Up).")
        self.fatigue_events = []   # 점수는 초기화 (챗봇 재발동 방지)
        self.sleep_trigger_count = 0 
        self.counter_ear = 0
        self.drowsiness_count += 1 # [신규] 재범 카운트 증가

    def process(self, frame, shape_np, face_rect):
        """
        메인 처리 로직
        Input: 프레임, 랜드마크(numpy), 얼굴 좌표
        Output: 현재 상태 문자열 (SAFE, DROWSY, SLEEP, YAWN, NO_FACE)
        """
        current_status = "SAFE"
        
        # ---------------------------------------------------------
        # [Step 0] 상태 유지 (Latching) 체크
        # ---------------------------------------------------------
        if time.time() - self.last_critical_time < self.CRITICAL_HOLD_TIME:
            # [수정] 얼굴이 감지되었을 때만 빨간 박스를 그립니다. (오류 방지)
            if face_rect is not None:
                cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            
            # 상태 텍스트는 얼굴 없어도 띄워줍니다.
            cv2.putText(frame, f"!!! {self.last_critical_status} !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.last_critical_status
            
        # 1. 얼굴 미검출 예외 처리
        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                return "NO_FACE"
            self.counter_ear = 0
            self.counter_mar = 0
            return "SAFE"
        self.counter_no_face = 0

        # ---------------------------------------------------------
        # [Step 1] Head Pose Estimation (고개 각도 계산)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # [Step 2] Smile Filter (웃음 오인식 방지)
        # ---------------------------------------------------------
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
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------------------------------------------------------
        # [Step 3] Yawn Detection (하품 감지)
        # ---------------------------------------------------------
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
                # [수정] 하품 시 current_status = "YAWN"으로 강제 변경하던 코드 삭제됨. (점수만 누적)
            elif self.counter_mar > self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.counter_mar = 0

        # ---------------------------------------------------------
        # [Step 4] Drowsiness Detection (EAR 기반 졸음 판단)
        # ---------------------------------------------------------
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        # [수정] 대화 중(is_speaking)일 때는 임계값을 2배로 확장
        target_frames = self.EAR_CONSEC_FRAMES * 2 if self.is_speaking else self.EAR_CONSEC_FRAMES

        if avg_ear < self.EAR_THRESHOLD:
            if not is_smile_mode and not is_yawning and not is_head_turned:
                self.counter_ear += 1
                
                # 유동적인 임계값(target_frames) 사용
                if self.counter_ear >= target_frames:
                    
                    # Case 1: 첫 번째 졸음 감지
                    if self.sleep_trigger_count == 0:
                        cv2.putText(frame, "!!! 1st WARNING !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        if self.counter_ear == target_frames:
                            self.sleep_trigger_count += 1

                        current_status = "DROWSY" 
                        self.last_critical_time = time.time()
                        self.last_critical_status = "DROWSY"

                    # Case 2: 두 번째 이후 감지 (Real Sleep)
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
                    # [수정] 대화 중이 아닐 때만 점수 누적
                    if not self.is_speaking:
                        self.add_fatigue_point(5, "Fast Blink") 
            self.counter_ear = 0

        # ---------------------------------------------------------
        # [Step 5] Total Score Evaluation (최종 상태 판정)
        # ---------------------------------------------------------
        total_score = self.get_current_fatigue_score()
        
        # 디버깅 정보 (화면 하단)
        info_text = f"Score: {total_score} | Count: {self.drowsiness_count} | Speak: {self.is_speaking}"
        cv2.putText(frame, info_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # [수정] 오직 점수가 80점 이상일 때만 DROWSY로 전환 (YAWN 상태 강제 제거됨)
        if current_status == "SAFE":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        color = (0, 255, 0)
        if current_status == "SLEEP": color = (0, 0, 255)       
        elif current_status == "DROWSY": color = (0, 165, 255) 
        
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), color, 2)
        
        return current_status

    def calibrate(self, frame, shape_np, face_rect):
        """
        초기 캘리브레이션 함수 (기존 코드 유지)
        """
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        progress = len(self.calib_ear_list)
        cv2.putText(frame, f"Progress: {progress}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

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
            print(f"✅ [Calibration Done] Eye Norm: {self.normal_eye_size:.3f} -> Thresh: {self.EAR_THRESHOLD:.3f}")