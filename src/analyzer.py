import cv2
import numpy as np
import time
import src.utils as utils

class DriverAnalyzer:
    """
    [운전자 상태 분석 엔진]
    - 역할: 프레임별 랜드마크 데이터를 받아 운전자의 상태(졸음, 하품, 부주의 등)를 진단
    - 특징:
        1. 개인화: 초기 캘리브레이션을 통해 사용자별 눈/입 크기 기준 설정
        2. 오탐 방지: 웃음(Smile), 대화(Speaking) 시 발생하는 눈 작아짐 현상을 졸음으로 오인하지 않도록 필터링
        3. 피로도 누적: 단순 순간 포착이 아닌, 최근 3분간의 누적 피로도(Score)로 상태 판정
    """

    def __init__(self):
        # ==========================================
        # [Configuration] 임계값 및 설정 상수
        # ==========================================
        # 1. 졸음 감지 (EAR: Eye Aspect Ratio)
        self.EAR_THRESHOLD = 0.25       # 눈 감김 판단 기준 (캘리브레이션 후 개인별 수치로 갱신됨)
        
        # [수정됨] 기존 60(2초) -> 30(1초)로 변경
        # 이제 1초만 눈을 감아도 졸음/수면으로 판정합니다.
        self.EAR_CONSEC_FRAMES = 30     # 졸음 판정 최소 지속 시간 (약 1초 @ 30fps)
        
        # 2. 하품 감지 (MAR: Mouth Aspect Ratio)
        self.MAR_THRESHOLD = 0.6        # 하품 판단 기준 (0.5에서 0.6으로 상향 조정하여 일반적인 입 벌림 제외)
        self.MAR_CONSEC_FRAMES = 30     # 하품 지속 시간 (약 1초)
        
        # 3. 고개 감지 (Head Pose Estimation)
        self.YAW_THRESHOLD = 40.0       # 고개 돌림 (전방 주시 태만) 각도
        self.NO_FACE_THRESHOLD = 50     # 얼굴이 카메라에서 사라졌을 때 경고하기까지의 허용 프레임
        self.CALIBRATION_FRAMES = 50    # 초기 기준값 설정을 위해 수집하는 샘플 프레임 수

        # 4. 스마트 필터 (Smart Filters)
        self.SMILE_RATIO = 1.3          # 평소 입 너비 대비 1.3배 이상 커지면 '웃음'으로 간주 (졸음 오탐 방지)
        self.SMILE_COOLDOWN = 3.0       # 웃은 뒤 3초간은 눈이 작아져도 졸음으로 판단하지 않음 (쿨다운)
        self.CRITICAL_HOLD_TIME = 3.0   # 위험 상태(졸음) 감지 시, 3초간 상태를 유지하여 UI에서 확인 가능하게 함 (Latching)

        # 5. 피로도 점수 시스템 (Fatigue Scoring)
        self.SCORE_WINDOW = 180         # 최근 3분(180초) 동안 쌓인 피로도만 계산 (Sliding Window)
        self.FATIGUE_THRESHOLD = 80     # 누적 점수가 80점을 넘으면 'DROWSY(졸음)' 상태 확정

        # ==========================================
        # [State] 상태 관리 변수
        # ==========================================
        self.is_calibrating = True      # 초기화 모드 플래그
        self.calib_ear_list = []        # 캘리브레이션용 EAR 데이터 버퍼
        self.calib_mouth_width_list = [] # 캘리브레이션용 입 너비 데이터 버퍼
        
        self.normal_eye_size = 0.30     # 사용자의 평소 눈 크기 (업데이트됨)
        self.normal_mouth_width = 0.0   # 사용자의 평소 입 너비 (업데이트됨)
        
        # 프레임 카운터 (지속 시간 측정용)
        self.counter_ear = 0
        self.counter_mar = 0
        self.counter_no_face = 0
        self.sleep_trigger_count = 0    # 1차 경고(Chatbot) -> 2차 비상(Alarm) 단계 추적용

        self.fatigue_events = []        # 피로도 이벤트 로그 [(timestamp, score, reason), ...]
        
        # [외부 연동 변수]
        self.drowsiness_count = 0       # 졸음 재범 횟수 (Track B: 상습 졸음 추적)
        self.is_speaking = False        # 챗봇과 대화 중인지 여부 (True일 경우 임계값 완화)

        self.last_smile_time = 0
        self.last_critical_time = 0
        self.last_critical_status = "SAFE"

        # ==========================================
        # [Camera & Model] 3D 좌표 계산용 모델
        # ==========================================
        # dlib 68 랜드마크 인덱스 정의
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # PnP Solver용 표준 3D 얼굴 모델 포인트 (Generic Face Model)
        # 2D 영상 좌표를 3D 공간 좌표로 변환하여 고개 각도를 계산하기 위함
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 코 끝 (Nose tip)
            (0.0, -330.0, -65.0),        # 턱 끝 (Chin)
            (-225.0, 170.0, -135.0),     # 왼쪽 눈 끝
            (225.0, 170.0, -135.0),      # 오른쪽 눈 끝
            (-150.0, -150.0, -125.0),    # 왼쪽 입꼬리
            (150.0, -150.0, -125.0)      # 오른쪽 입꼬리
        ], dtype="double")
        
        # 카메라 매트릭스 (RPi Camera V3 Wide 렌즈 왜곡을 고려한 추정치)
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def get_mouth_aspect_ratio(self, mouth):
        """입의 벌림 정도(MAR) 계산: 세로 높이 / 가로 너비"""
        A = np.linalg.norm(mouth[2] - mouth[10]) # 세로 1
        B = np.linalg.norm(mouth[4] - mouth[8])  # 세로 2
        C = np.linalg.norm(mouth[0] - mouth[6])  # 가로
        return (A + B) / (2.0 * C)

    def add_fatigue_point(self, score, reason):
        """
        [피로도 점수 누적]
        - 점수 추가 시 '말하는 중(is_speaking)'인지 확인하여 억울한 점수 누적을 방지함.
        """
        if not self.is_speaking:
            self.fatigue_events.append((time.time(), score, reason))

    def get_current_fatigue_score(self):
        """
        [현재 피로도 계산]
        - 최근 3분(SCORE_WINDOW) 이내의 이벤트 점수만 합산 (Sliding Window)
        """
        current_time = time.time()
        # 유효 기간 지난 점수 제거
        self.fatigue_events = [evt for evt in self.fatigue_events if current_time - evt[0] < self.SCORE_WINDOW]
        return sum(evt[1] for evt in self.fatigue_events)

    def refund_blink_points(self):
        """
        [점수 환불 시스템]
        - 웃음(Smile)이 감지되면, 최근 2초간 '빠른 눈 깜빡임'으로 잘못 기록된 점수를 무효화 처리함.
        """
        current_time = time.time()
        self.fatigue_events = [
            evt for evt in self.fatigue_events
            if not ((current_time - evt[0] < 2.0) and (evt[2] == "Fast Blink"))
        ]

    def reset_full_calibration(self):
        """
        [완전 초기화]
        - 사용자가 졸음을 부정(DENY)하거나 운전자가 바뀌었을 때 실행
        - 점수, 카운트, 기준값 등 모든 데이터를 초기화하고 재보정 모드로 진입
        """
        print("🔄 [Analyzer] System Full Reset (Score + Count).")
        self.is_calibrating = True
        self.calib_ear_list = []
        self.calib_mouth_width_list = []
        self.sleep_trigger_count = 0  
        self.counter_ear = 0
        self.fatigue_events = []
        self.drowsiness_count = 0  # 재범 카운트까지 초기화

    def reset_soft_for_next_stage(self):
        """
        [부분 초기화]
        - 사용자가 졸음을 인정(ADMIT)하거나 경고가 발동되었을 때 실행
        - 누적 점수는 비우되, 재범 카운트(drowsiness_count)는 증가시켜 다음 감지 시 대응 단계를 높임
        """
        print("⚠️ [Analyzer] Soft Reset (Count Up).")
        self.fatigue_events = []   # 점수는 초기화 (챗봇 중복 발동 방지)
        self.sleep_trigger_count = 0 
        self.counter_ear = 0
        self.drowsiness_count += 1 # 재범 횟수 증가 (경고 -> 비상벨 격상 근거)

    def process(self, frame, shape_np, face_rect):
        """
        [메인 분석 파이프라인]
        Input: 프레임 이미지, 랜드마크 좌표(numpy), 얼굴 박스
        Output: 현재 상태 (SAFE, DROWSY, SLEEP, YAWN, NO_FACE)
        """
        current_status = "SAFE"
        
        # ---------------------------------------------------------
        # [Step 0] 상태 유지 (Latching)
        # 경고 문구가 너무 빨리 사라지지 않도록 일정 시간(3초) 화면에 고정 표시
        # ---------------------------------------------------------
        if time.time() - self.last_critical_time < self.CRITICAL_HOLD_TIME:
            # 얼굴 박스 그리기 (오류 방지: 얼굴이 감지된 상태에서만)
            if face_rect is not None:
                cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 0, 255), 2)
            
            # 상태 텍스트 출력
            cv2.putText(frame, f"!!! {self.last_critical_status} !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return self.last_critical_status
            
        # 1. 얼굴 미검출(No Face) 예외 처리
        if shape_np is None:
            self.counter_no_face += 1
            if self.counter_no_face >= self.NO_FACE_THRESHOLD:
                return "NO_FACE"
            # 얼굴을 잠깐 놓친 경우엔 카운터 리셋하며 대기
            self.counter_ear = 0
            self.counter_mar = 0
            return "SAFE"
        self.counter_no_face = 0

        # ---------------------------------------------------------
        # [Step 1] Head Pose Estimation (고개 각도 계산)
        # PnP(Perspective-n-Point) 알고리즘을 이용해 얼굴의 3D 방향 벡터 추출
        # ---------------------------------------------------------
        img_pts = np.array([shape_np[30], shape_np[8], shape_np[36], shape_np[45], shape_np[48], shape_np[54]], dtype="double")
        (_, rvec, tvec) = cv2.solvePnP(self.model_points, img_pts, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (rmat, _) = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_mat)[6]
        try: yaw = euler[1][0] # Y축 회전 (좌우 도리도리)
        except: yaw = 0
        
        # 고개를 너무 많이 돌리면 부주의(Side Look)로 간주
        is_head_turned = abs(yaw) > self.YAW_THRESHOLD  
        if is_head_turned:
            cv2.putText(frame, f"SIDE LOOK ({int(yaw)})", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ---------------------------------------------------------
        # [Step 2] Smile Filter (웃음 오인식 방지)
        # 웃을 때는 눈이 작아지고 입꼬리가 올라가므로, 이를 졸음으로 오판하지 않도록 필터링
        # ---------------------------------------------------------
        mouth_pts = shape_np[self.MOUTH]
        current_mouth_width = utils.get_mouth_width(mouth_pts)
        
        is_smiling_now = False
        if self.normal_mouth_width > 0:
            # 평소 입 너비보다 1.3배 이상 길어지면 웃음으로 판단
            if current_mouth_width > self.normal_mouth_width * self.SMILE_RATIO:
                is_smiling_now = True
                self.last_smile_time = time.time()
                self.refund_blink_points() # 웃느라 작아진 눈 때문에 쌓인 점수 환불

        # 웃음 쿨다운 적용 (웃은 직후 3초간은 졸음 판단 보류)
        is_smile_mode = is_smiling_now or (time.time() - self.last_smile_time < self.SMILE_COOLDOWN)

        if is_smile_mode:
            cv2.putText(frame, "^^ SMILING ^^", (face_rect.left(), face_rect.top()-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ---------------------------------------------------------
        # [Step 3] Yawn Detection (하품 감지)
        # ---------------------------------------------------------
        mouth_center = np.mean(mouth_pts, axis=0)
        # 시각화용 입술 외곽선 그리기
        mouth_pts_visual = (mouth_pts - mouth_center) * [0.95, 0.60] + mouth_center
        cv2.drawContours(frame, [cv2.convexHull(mouth_pts_visual.astype(np.int32))], -1, (0, 255, 255), 1)
        
        mar = self.get_mouth_aspect_ratio(mouth_pts)
        is_yawning = False
        
        # 웃거나 말하는 중이 아닐 때만 하품 검사
        if mar > self.MAR_THRESHOLD and not is_smile_mode and not self.is_speaking:
            is_yawning = True 
            self.counter_mar += 1
            if self.counter_mar == self.MAR_CONSEC_FRAMES: 
                self.add_fatigue_point(30, "Yawn") # 하품 감지 시 피로도 점수 +30
            elif self.counter_mar > self.MAR_CONSEC_FRAMES:
                cv2.putText(frame, "!!! YAWNING !!!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            self.counter_mar = 0

        # ---------------------------------------------------------
        # [Step 4] Drowsiness Detection (EAR 기반 졸음 판단)
        # ---------------------------------------------------------
        left_pts = shape_np[self.LEFT_EYE]
        right_pts = shape_np[self.RIGHT_EYE]
        # 양쪽 눈의 평균 EAR 계산
        avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0

        # 적응형 임계값: 눈을 크게 뜰 때마다 기준값을 미세하게 상향 조정하여 정확도 유지
        if avg_ear > self.EAR_THRESHOLD:
            self.normal_eye_size = (self.normal_eye_size * 0.99) + (avg_ear * 0.01)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

        # 눈 시각화 (초록색 테두리)
        cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

        # [핵심] 대화 중(is_speaking)일 때는 눈을 자주 깜빡이거나 가늘게 뜨므로 허용 시간 2배 연장
        target_frames = self.EAR_CONSEC_FRAMES * 2 if self.is_speaking else self.EAR_CONSEC_FRAMES

        if avg_ear < self.EAR_THRESHOLD:
            # 웃음, 하품, 고개 돌림이 아닌 순수 졸음 상황일 때
            if not is_smile_mode and not is_yawning and not is_head_turned:
                self.counter_ear += 1
                
                # 지속 시간이 기준치(target_frames)를 초과하면 졸음 판정
                if self.counter_ear >= target_frames:
                    
                    # Case 1: 첫 번째 졸음 감지 (경고 단계)
                    if self.sleep_trigger_count == 0:
                        cv2.putText(frame, "!!! 1st WARNING !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
                        if self.counter_ear == target_frames:
                            self.sleep_trigger_count += 1 # 상태 격상

                        current_status = "DROWSY" 
                        self.last_critical_time = time.time()
                        self.last_critical_status = "DROWSY"

                    # Case 2: 두 번째 이후 감지 (수면 단계 - 비상)
                    else:
                        cv2.putText(frame, "!!! SLEEP !!!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if self.counter_ear == target_frames:
                            self.add_fatigue_point(60, "Long Blink") # 강한 졸음 점수 +60
                        
                        current_status = "SLEEP"
                        self.last_critical_time = time.time()
                        self.last_critical_status = "SLEEP"

        else:
            # 눈을 떴을 때
            if not is_smile_mode and not is_yawning and not is_head_turned:
                # 짧은 깜빡임(Fast Blink)도 너무 잦으면 피로 징후로 간주하여 소량 점수 누적
                if self.counter_ear > 5 and self.counter_ear < target_frames:
                    if not self.is_speaking: # 대화 중에는 깜빡임 점수 면제
                        self.add_fatigue_point(5, "Fast Blink") 
            self.counter_ear = 0

        # ---------------------------------------------------------
        # [Step 5] Total Score Evaluation (최종 상태 판정)
        # ---------------------------------------------------------
        total_score = self.get_current_fatigue_score()
        
        # 디버깅 정보 출력 (점수, 재범 횟수, 대화 상태)
        info_text = f"Score: {total_score} | Count: {self.drowsiness_count} | Speak: {self.is_speaking}"
        cv2.putText(frame, info_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # [최종 판단] 눈을 뜨고 있어도 누적 피로도가 80점을 넘으면 'DROWSY' 상태로 강제 전환
        if current_status == "SAFE":
            if total_score >= self.FATIGUE_THRESHOLD:
                current_status = "DROWSY"

        # 상태별 얼굴 박스 색상 변경 (초록 -> 주황 -> 빨강)
        color = (0, 255, 0)
        if current_status == "SLEEP": color = (0, 0, 255)       
        elif current_status == "DROWSY": color = (0, 165, 255) 
        
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), color, 2)
        
        return current_status

    def calibrate(self, frame, shape_np, face_rect):
        """
        [캘리브레이션 모드]
        - 시스템 시작 시 사용자에게 정면을 보게 하여 평소 눈/입 크기를 측정
        """
        cv2.putText(frame, "CALIBRATING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT THE MONITOR", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        progress = len(self.calib_ear_list)
        cv2.putText(frame, f"Progress: {progress}/{self.CALIBRATION_FRAMES}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if shape_np is not None:
            left_pts = shape_np[self.LEFT_EYE]
            right_pts = shape_np[self.RIGHT_EYE]
            avg_ear = (utils.get_eye_aspect_ratio(left_pts) + utils.get_eye_aspect_ratio(right_pts)) / 2.0
            
            # 눈을 감고 있는 프레임(EAR < 0.15)은 평균 계산에서 제외
            if avg_ear > 0.15: 
                self.calib_ear_list.append(avg_ear)
            
            mouth_pts = shape_np[self.MOUTH]
            mouth_width = utils.get_mouth_width(mouth_pts)
            self.calib_mouth_width_list.append(mouth_width)

            cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 255), 2)

        # 데이터 수집 완료 시
        if len(self.calib_ear_list) >= self.CALIBRATION_FRAMES:
            self.calib_ear_list.sort()
            
            # 상위 90% 값을 평소 눈 크기로 설정 (최댓값보다는 안정적)
            index_90th = int(len(self.calib_ear_list) * 0.90)
            self.normal_eye_size = self.calib_ear_list[index_90th]
            
            # 졸음 감지 임계값 설정 (평소 크기의 85% 이하로 떨어지면 졸음)
            self.EAR_THRESHOLD = max(0.20, self.normal_eye_size * 0.85)

            if self.calib_mouth_width_list:
                self.calib_mouth_width_list.sort()
                mid_index = len(self.calib_mouth_width_list) // 2
                self.normal_mouth_width = self.calib_mouth_width_list[mid_index]
            
            self.is_calibrating = False
            self.sleep_trigger_count = 0 
            print(f"✅ [Calibration Done] Eye Norm: {self.normal_eye_size:.3f} -> Thresh: {self.EAR_THRESHOLD:.3f}")