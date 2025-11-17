import cv2
from picamera2 import Picamera2

print("OpenCV version:", cv2.__version__)
print("Starting Picamera2... 'q' to quit.")

# 1. Picamera2 객체 생성
picam2 = Picamera2()

# 2. 카메라 설정: 640x480 해상도의 프리뷰 설정
#    (논문에서처럼 해상도를 낮추는 것이 FPS에 유리합니다)
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

# 3. 카메라 시작
picam2.start()

while True:
    # 4. 카메라에서 프레임(이미지) 캡처 (NumPy 배열로 바로 가져옴)
    frame = picam2.capture_array()

    # 5. OpenCV로 창에 보여주기 (이 부분은 동일)
    #    (참고: PiCamera는 색상이 BGR이 아닌 RGB 순서일 수 있습니다.
    #     만약 색이 이상하게 보이면 frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     코드를 추가해야 할 수도 있습니다. 일단 테스트!)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Picamera2 Test", frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 리소스 정리
cv2.destroyAllWindows()
picam2.stop()
print("테스트 종료.")