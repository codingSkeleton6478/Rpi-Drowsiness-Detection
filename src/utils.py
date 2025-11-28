import numpy as np
import cv2

def euclidean_dist(ptA: np.ndarray, ptB: np.ndarray) -> float:
    """
    두 점(Numpy Array) 사이의 유클리드 거리(L2 Norm)를 계산합니다.
    """
    return np.linalg.norm(ptA - ptB)

def get_eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    눈의 종횡비(EAR, Eye Aspect Ratio)를 계산합니다.
    - 논문: Soukupová & Čech (2016)
    
    Args:
        eye_points: 6개의 (x, y) 좌표를 담은 numpy 배열 (dlib 36~41 or 42~47)
        
    Returns:
        float: 계산된 EAR 값
    """
    # 1. 수직 눈꺼풀 사이의 거리 (Vertical)
    # P2(1) <-> P6(5)
    A = euclidean_dist(eye_points[1], eye_points[5])
    # P3(2) <-> P5(4)
    B = euclidean_dist(eye_points[2], eye_points[4])

    # 2. 수평 눈 끝 사이의 거리 (Horizontal)
    # P1(0) <-> P4(3)
    C = euclidean_dist(eye_points[0], eye_points[3])

    # 3. EAR 계산
    # 분모(C)가 0일 경우를 대비해 아주 작은 값(epsilon)을 더함 (ZeroDivisionError 방지)
    ear = (A + B) / (2.0 * (C + 1e-6))

    return ear

def get_mouth_width(mouth_points: np.ndarray) -> float:
    """
    입의 가로 너비(Width)를 계산합니다.
    
    Args:
        mouth_points: dlib 랜드마크 48~67번 (총 20개 점)
        - Index 0: 48번 (왼쪽 입꼬리)
        - Index 6: 54번 (오른쪽 입꼬리)
    """
    # 48번(Left Corner)과 54번(Right Corner) 사이의 거리
    return euclidean_dist(mouth_points[0], mouth_points[6])

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    이미지의 명암(Gamma)을 조절하여 조명 변화에 대응합니다.
    
    Args:
        image: 입력 이미지 (OpenCV BGR)
        gamma: 감마 값 (1.0=원본, <1.0=어둡게, >1.0=밝게)
               * 야간 운전 시 1.5~2.0 권장
               
    Returns:
        보정된 이미지 (Numpy Array)
    """
    # 감마가 1.0이면 연산 없이 원본 반환 (성능 최적화)
    if gamma == 1.0:
        return image

    invGamma = 1.0 / gamma
    
    # 룩업 테이블(LUT) 생성: 픽셀마다 계산하지 않고 미리 계산된 표를 사용
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # LUT 적용 (고속 연산)
    return cv2.LUT(image, table)