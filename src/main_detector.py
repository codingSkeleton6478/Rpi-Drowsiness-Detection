import cv2
import dlib
import time
from picamera2 import Picamera2

print("OpenCV version:", cv2.__version__)
print("Dlib version:", dlib.__version__)
print("Starting detector... 'q' to quit.")

# 1. Dlib의 얼굴 탐지기 및 랜드마크 예측기 로드
# ----------------------------------------------------------------
# (주의!) 이 스크립트는 'cpst' 폴더에서 실행해야 경로가 맞습니다.
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# ----------------------------------------------------------------

# 2. Picamera2 객체 생성 및 설정
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# FPS 계산용 변수
prev_time = 0

while True:
    # 3. 카메라 프레임 캡처
    frame = picam2.capture_array()

    # 4. 색상 보정 (RGB -> BGR) - imshow용
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 5. Dlib 감지를 위한 그레이스케일 이미지 생성
    #    (컬러 이미지를 BGR로 바꾼 frame_bgr에서 변환)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 6. 그레이스케일 이미지에서 얼굴 탐지
    faces = detector(gray, 0)

    # 7. 탐지된 얼굴들에 대해 랜드마크 찾기
    for face in faces:
        # 얼굴 영역에 사각형 그리기 (B, G, R), 두께
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ⭐️ [수정됨] 랜드마크(68개 점)를 찾아 빨간색 점으로 그리기
        shape = predictor(gray, face)
        for i in range(0, 68):
            (px, py) = (shape.part(i).x, shape.part(i).y)
            cv2.circle(frame_bgr, (px, py), 1, (0, 0, 255), -1)

    # 8. FPS 계산 및 표시
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 9. 화면에 결과 표시 (창 제목 변경)
    cv2.imshow("Face & Landmark Detector", frame_bgr)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. 리소스 정리
cv2.destroyAllWindows()
picam2.stop()
print("Detector 종료.")