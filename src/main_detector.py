import cv2
import dlib
import time
import numpy as np
from picamera2 import Picamera2
import utils # 'src.' 제거 확인

print("OpenCV version:", cv2.__version__)
print("Dlib version:", dlib.__version__)
print("Starting detector... 'q' to quit.")

# 1. [수정] 졸음 판단 기준 상수 정의
# -----------------------------------------------------------------
# EAR 임계값 (이 값보다 낮으면 "눈 감음"으로 간주)
EAR_THRESHOLD = 0.2
# EAR 임계값 미만으로 지속되어야 하는 최소 연속 프레임 수 (FPS에 따라 조절)
# 예: FPS가 20~25 정도라면, 40 프레임은 약 2초에 해당함
EAR_CONSEC_FRAMES = 40 
# -----------------------------------------------------------------

# 2. [수정] 눈 감은 프레임 카운터 및 졸음 상태 플래그
COUNTER = 0
DROWSY = False # 졸음 상태 여부

# 3. Dlib 탐지기 및 랜드마크 인덱스 정의
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))

# 4. dlib shape을 numpy 배열로 변환하는 함수
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# 5. Picamera2 객체 생성 및 설정
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# FPS 계산용 변수
prev_time = 0

while True:
    # 6. 카메라 프레임 캡처 및 기본 처리
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 7. 얼굴 탐지
    faces = detector(gray, 0)
    
    # [수정] 얼굴이 감지되지 않았을 때를 대비해 DROWSY 플래그 초기화
    if len(faces) == 0:
        DROWSY = False
        COUNTER = 0

    # 8. 탐지된 얼굴들에 대해 루프
    for face in faces:
        # 얼굴 영역 사각형
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 랜드마크 탐지 및 numpy 배열로 변환
        shape = predictor(gray, face)
        shape_np = shape_to_np(shape)

        # 9. EAR 계산
        left_eye_pts = shape_np[LEFT_EYE_INDICES]
        right_eye_pts = shape_np