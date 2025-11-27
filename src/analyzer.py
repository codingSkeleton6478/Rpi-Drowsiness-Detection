class DriverAnalyzer:
    def __init__(self):
        # ==========================================
        # 사용자 원본 설정값 복구 (고정값 사용)
        # ==========================================
        self.EAR_THRESHOLD = 0.25       # 눈 감음 기준
        self.EAR_CONSEC_FRAMES = 15     # 졸음 프레임 (약 0.5~1초)
        self.SLEEP_CONSEC_FRAMES = 30   # 수면 프레임 (약 1.5초 이상)
        
        self.MAR_THRESHOLD = 0.5        # 하품 기준
        self.MAR_CONSEC_FRAMES = 10
        
        self.PITCH_THRESHOLD = 20.0     # 고개 숙임
        
        self.counter_ear = 0
        self.counter_mar = 0
        
        # 캘리브레이션 기능 (필요시 'r'키로 작동하게 남겨둠)
        self.is_calibrating = False
        self.calib_ear_list = []

    def process(self, data):
        if data is None:
            self.counter_ear = 0
            return "NO_FACE", {}

        ear = data["ear"]
        mar = data["mar"]
        pitch = data["pitch"]
        
        status = "SAFE"

        # 1. 캘리브레이션 (옵션)
        if self.is_calibrating:
            if ear > 0.15: self.calib_ear_list.append(ear)
            if len(self.calib_ear_list) > 50:
                avg_ear = sum(self.calib_ear_list) / len(self.calib_ear_list)
                self.EAR_THRESHOLD = avg_ear * 0.85 # 평소 눈 크기의 85%로 설정
                self.is_calibrating = False
                print(f"✅ 캘리브레이션 완료! New Threshold: {self.EAR_THRESHOLD:.3f}")
            return "CALIBRATING", {"progress": len(self.calib_ear_list)}

        # 2. 졸음/수면 판단 (카운터 방식)
        if ear < self.EAR_THRESHOLD:
            self.counter_ear += 1
        else:
            self.counter_ear = 0

        if mar > self.MAR_THRESHOLD:
            self.counter_mar += 1
        else:
            self.counter_mar = 0

        # 상태 결정 (심각도 순서: 수면 > 졸음/하품 > 정상)
        if self.counter_ear >= self.SLEEP_CONSEC_FRAMES:
            status = "CRITICAL_SLEEP"
        elif pitch > self.PITCH_THRESHOLD:
            status = "HEAD_DOWN"
        elif self.counter_ear >= self.EAR_CONSEC_FRAMES:
            status = "DROWSY"
        elif self.counter_mar >= self.MAR_CONSEC_FRAMES:
            status = "YAWN"

        return status, {
            "ear": ear, "mar": mar, "ear_thresh": self.EAR_THRESHOLD
        }