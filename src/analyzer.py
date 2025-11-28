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
        self.MAR_THRESHOLD = 0.5        # 입 벌림 기준
        self.MAR_CONSEC_FRAMES = 30     # 약 1초 지속 시 하품 판정
        
        # 3. 고개 감지 (Head Pose)
        self.PITCH_THRESHOLD = 20.0     # 고개 숙임 (Sleeping)
        self.YAW_THRESHOLD = 40.0       # 고개 돌림 (Distraction)
        self.NO_FACE_THRESHOLD = 50     # 얼굴 소실 허용 프레임 수
        self.CALIBRATION_FRAMES = 50    # 캘리브레이션에 필요한 프레임 수

        # 4. 스마트 필터 (웃음 & 상태 유지)

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

    def reset_calibration(self):
        """사용자 변경 또는 재설정 시 캘리브레이션 초기화"""
        print("🔄 [Analyzer] System Reset & Calibration Started.")
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        self.sleep_trigger_count = 0  
        self.counter_ear = 0
        self.fatigue_events = []

    def process(self, frame, shape_np, face_rect):
        """
        메인 처리 로직
        Input: 프레임, 랜드마크(numpy), 얼굴 좌표
        Output: 현재 상태 문자열 (SAFE, DROWSY, SLEEP, YAWN, NO_FACE)
        """
        current_status = "SAFE"
        
        # ---------------------------------------------------------
        # [Step 0] 상태 유지 (Latching) 체크
        # 위험 상태였다면 즉시 해제하지 않고 CRITICAL_HOLD_TIME 동안 유지
        # ---------------------------------------------------------
        if time.time() - self.last_critical_time < self.CRITICAL_HOLD_TIME:
            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, f"!!! {self.last_critical_status} !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.last_critical_status

        # 1. 얼굴 미검출 예외 처리
        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                cv2.putText(frame, "!!! FACE LOST !!!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
        
        # 고개 돌림 감지 (부주의)
        is_head_turned = abs(yaw) > self.YAW_THRESHOLD  
        if is_head_turned:
            cv2.putText(frame, f"SIDE LOOK ({int(yaw)})", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ---------------------------------------------------------
        # [Step 2] Smile Filter (웃음 오인식 방지)
        # 입의 너비가 평소보다 넓어지면 웃음으로 판단 -> 졸음 감지 일시 중지
        # ---------------------------------------------------------
        mouth_pts = shape_np[self.MOUTH]
        current_mouth_width = utils.get_mouth_width(mouth_pts)
        
        is_smiling_now = False
        if self.normal_mouth_width > 0:
            if current_mouth_width > self.normal_mouth_width * self.SMILE_RATIO:
                is_smiling_now = True
                self.last_smile_time = time.time()
                self.refund_blink_points() # 웃느라 눈 작아진 건 점수 환불

        # 쿨다운 적용: 웃음이 끝났어도 잠시동안은 졸음 판단 보류
        is_smile_mode = is_smiling_now or (time.time() - self.last_smile_time < self.SMILE_COOLDOWN)

        if is_smile_mode:
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------------------------------------------------------
        # [Step 3] Yawn Detection (하품 감지)
        # ---------------------------------------------------------
        # 입술 외곽선 시각화
        mouth_center = np.mean(mouth_pts, axis=0)
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.60] + mouth_center
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual.astype(np.int32))], -1, (0, 255, 255), 1)
        
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        is_yawning = False
        
        if mar > self.MAR_THRESHOLD and not is_smile_mode:
            is_yawning = True 
            self.counter_mar += 1
            if self.counter_mar == self.MAR_CONSEC_FRAMES: 
                self.add_fatigue_point(30, "Yawn") 
                if current_status == "SAFE": current_status = "YAWN"
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

        # Adaptive Threshold: 눈을 크게 뜨면 기준값도 서서히 올라감 (적응형)
        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        # 눈 시각화
        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        # 눈 감김 체크 (웃음, 하품, 고개돌림 상황 제외)
        if avg_ear < self.EAR_THRESHOLD:
            if not is_smile_mode and not is_yawning and not is_head_turned:
                self.counter_ear += 1
                
                # 2초 이상 지속 (SLEEP 조건 충족)
                if self.counter_ear >= self.EAR_CONSEC_FRAMES:
                    
                    # Case 1: 첫 번째 졸음 감지 (1st Warning -> 챗봇 유도)
                    if self.sleep_trigger_count == 0:
                        cv2.putText(frame, "!!! 1st WARNING !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        if self.counter_ear == self.EAR_CONSEC_FRAMES:
                            self.sleep_trigger_count += 1
                            # 1차 경고는 점수 부여 안 함 (오탐 가능성)

                        current_status = "DROWSY" 
                        self.last_critical_time = time.time()
                        self.last_critical_status = "DROWSY"

                    # Case 2: 두 번째 이후 감지 (Real Sleep -> 강력 경고)
                    else:
                        cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if self.counter_ear == self.EAR_CONSEC_FRAMES:
                            self.add_fatigue_point(60, "Long Blink") # 점수 대폭 추가
                        
                        current_status = "SLEEP"
                        self.last_critical_time = time.time()
                        self.last_critical_status = "SLEEP"

        else:
            # 눈 떴을 때: 짧은 깜빡임(Fast Blink) 패턴 분석
            if not is_smile_mode and not is_yawning and not is_head_turned:
                # 너무 짧지도(노이즈), 너무 길지도(졸음) 않은 깜빡임은 피로도로 누적
                if self.counter_ear > 5 and self.counter_ear < self.EAR_CONSEC_FRAMES:
                    self.add_fatigue_point(5, "Fast Blink") 
            self.counter_ear = 0

        # ---------------------------------------------------------
        # [Step 5] Total Score Evaluation (최종 상태 판정)
        # ---------------------------------------------------------
        total_score = self.get_current_fatigue_score()
        cv2.putText(frame, f"Fatigue Score: {total_score}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 졸음이나 하품 상태가 아니더라도 점수가 높으면 DROWSY 강제
        if current_status == "SAFE" or current_status == "YAWN":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        # 결과 박스 그리기
        color = (0, 255, 0) # Green (SAFE)
        if current_status == "SLEEP": color = (0, 0, 255)       # Red
        elif current_status in ["DROWSY", "YAWN"]: color = (0, 165, 255) # Orange
        
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), color, 2)
        
        return current_status

    def calibrate(self, frame, shape_np, face_rect):
        """
        초기 캘리브레이션 함수
        - 목적: 사용자별 눈 크기(EAR)와 입 너비 표준값 측정
        - 방식: CALIBRATION_FRAMES 동안 데이터를 수집하여 상위 90% 값을 기준점으로 설정
        """
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        progress = len(self.calib_ear_list)
        cv2.putText(frame, f"Progress: {progress}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if shape_np is not None:
            left_pts = shape_np[self.LEFT_EYE]
            right_pts = shape_np[self.RIGHT_EYE]
            avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0
            
            # 눈을 너무 작게 뜬 데이터는 노이즈로 보고 제외 (0.15 이상만 수집)
            if avg_ear > 0.15: 
                self.calib_ear_list.append(avg_ear)
            
            mouth_pts = shape_np[self.MOUTH]
            mouth_width = utils.get_mouth_width(mouth_pts)
            self.calib_mouth_width_list.append(mouth_width)

            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 255), 2)

        # 데이터 수집 완료 시 분석
        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            
            # 눈 크기: 상위 90% 지점을 '평소 눈 크기'로 설정 (눈을 깜빡인 데이터 제외 목적)
            index_90th = int(len(self.calib_ear_list) * 0.90)
            self.normal_eye_size = self.calib_ear_list[index_90th]
            
            # 임계값 설정: 평소 눈 크기의 85% 이하로 떨어지면 졸음으로 간주 (최소값 0.20 보장)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

            if self.calib_mouth_width_list:
                self.calib_mouth_width_list.sort()
                mid_index = len(self.calib_mouth_width_list) // 2
                self.normal_mouth_width = self.calib_mouth_width_list[mid_index]
            
            self.is_calibrating = False
            self.sleep_trigger_count = 0 
            print(f"✅ [Calibration Done] Eye Norm: {self.normal_eye_size:.3f} -> Thresh: {self.EAR_THRESHOLD:.3f}")