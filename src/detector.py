import cv2
import dlib
import numpy as np
import src.utils as utils

class FaceDetector:
    def __init__(self):
        print("✅ [Detector] Dlib 모델 로드 중 (원본 설정)...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        
        # 랜드마크 인덱스
        self.LEFT_EYE = list(range(36, 42))
        self.RIGHT_EYE = list(range(42, 48))
        self.MOUTH = list(range(48, 68))
        
        # Head Pose용 3D 모델 좌표
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype="double")
        
        self.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4,1))

    def expand_rect(self, rect, frame_shape):
        img_h, img_w = frame_shape[:2]
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        pad_top = int(h * 0.05)
        pad_bottom = int(h * 0.25) 
        pad_side = int(w * 0.15)
        
        new_x1 = max(0, x - pad_side)
        new_y1 = max(0, y - pad_top)
        new_x2 = min(img_w, x + w + pad_side)
        new_y2 = min(img_h, y + h + pad_bottom)
        
        return dlib.rectangle(new_x1, new_y1, new_x2, new_y2)

    def get_landmarks(self, gray, face):
        shape = self.predictor(gray, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    def get_mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def detect(self, frame, gray):
        """
        [핵심] main.py에서 호출하는 함수 이름은 'detect' 입니다.
        얼굴을 찾고 랜드마크(shape)와 얼굴박스(rect)를 반환합니다.
        """
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None, None

        # 가장 큰 얼굴 하나만 처리
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        expanded_face = self.expand_rect(face, frame.shape)

        try:
            shape_np = self.get_landmarks(gray, expanded_face)
            return shape_np, expanded_face
        except:
            return None, None