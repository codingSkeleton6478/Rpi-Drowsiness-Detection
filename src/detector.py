import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self):
        print("✅ [Detector] Dlib 모델 로드 중 (수치 튜닝: 상하10/좌우0)...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def expand_rect(self, rect, frame_shape):
        """
        [수치 수정됨] 
        - Top: 10% (유지)
        - Bottom: 10% (기존 20% -> 10%로 축소: 위아래 균형 맞춰서 눈 위치 교정)
        - Side: 0% (기존 10% -> 0%로 제거: 입술 비대화 해결)
        """
        img_h, img_w = frame_shape[:2]
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        
        # 1. 위쪽(Top): 10% (유지)
        pad_top = int(h * 0.10)
        
        # 2. 아래쪽(Bottom): 10% (수정됨)
        pad_bottom = int(h * 0.10) 
        
        # 3. 좌우(Side): 0% (수정됨 - 패딩 제거)
        pad_side = int(w * 0.00)
        
        new_x1 = max(0, x - pad_side)
        new_y1 = max(0, y - pad_top)
        new_x2 = min(img_w, x + w + pad_side)
        new_y2 = min(img_h, y + h + pad_bottom)
        
        return dlib.rectangle(new_x1, new_y1, new_x2, new_y2)

    def get_landmarks(self, gray, face):
        """Dlib 객체를 Numpy 배열로 변환"""
        shape = self.predictor(gray, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    def detect(self, frame, gray):
        """
        얼굴을 찾고 [랜드마크 좌표 배열]과 [얼굴 박스]를 반환
        """
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None, None

        # 가장 큰 얼굴 하나만 처리
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # 튜닝된 비율로 박스 확장
        expanded_face = self.expand_rect(face, frame.shape)

        try:
            # 랜드마크 추출 및 Numpy 변환
            shape_np = self.get_landmarks(gray, expanded_face)
            return shape_np, expanded_face
        except:
            # 만약 확장된 박스에서 실패하면 원본 박스로 재시도 (안전장치)
            try:
                shape_np = self.get_landmarks(gray, face)
                return shape_np, face
            except:
                return None, None