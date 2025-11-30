import cv2
import dlib
import numpy as np

class FaceDetector:
    def __init__(self):
        # [모델 초기화] Dlib의 HOG 기반 얼굴 감지기 및 68 포인트 랜드마크 예측기 로드
        print("✅ [Detector] Dlib 모델 로드 중 (수치 튜닝: 상하10/좌우0)...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    def expand_rect(self, rect, frame_shape):
        """
        [ROI 영역 확장 함수]
        Dlib이 감지한 얼굴 박스가 너무 타이트하여 눈/입 랜드마크가 경계에 걸리는 문제를 방지합니다.
        
        [수치 튜닝 히스토리] 
        - Top: 10% (유지) -> 이마 영역 확보
        - Bottom: 10% (기존 20% -> 10% 축소) -> 위아래 균형을 맞춰 눈의 상대적 위치(PnP Solver) 오차 감소
        - Side: 0% (기존 10% -> 0% 제거) -> 측면 확장 시 배경 노이즈로 인해 입술이 비대하게 인식되는 오류(Lip Hypertrophy) 해결
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
        """Dlib의 랜드마크 객체를 연산 효율성을 위해 Numpy 배열로 변환"""
        shape = self.predictor(gray, face)
        return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype="int")

    def detect(self, frame, gray):
        """
        [얼굴 감지 메인 메서드]
        프레임에서 얼굴을 찾고, 튜닝된 박스를 기반으로 랜드마크를 추출합니다.
        """
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None, None

        # [단일 사용자 집중] 여러 얼굴이 감지될 경우, 가장 크기가 큰(가까운) 얼굴 하나만 처리
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # [전처리] 랜드마크 안정성을 위해 박스 확장 적용
        expanded_face = self.expand_rect(face, frame.shape)

        try:
            # 1차 시도: 확장된 박스 기준으로 랜드마크 추출
            shape_np = self.get_landmarks(gray, expanded_face)
            return shape_np, expanded_face
        except:
            # [Fail-safe] 예외 처리
            # 확장된 박스가 이미지 범위를 벗어나거나 오류 발생 시, 원본 박스로 재시도 (안전장치)
            try:
                shape_np = self.get_landmarks(gray, face)
                return shape_np, face
            except:
                return None, None