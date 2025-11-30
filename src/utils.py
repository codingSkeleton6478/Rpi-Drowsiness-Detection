import numpy as np
import cv2

def euclidean_dist(ptA: np.ndarray, ptB: np.ndarray) -> float:
    """
    [유클리드 거리 계산]
    - 두 점(Numpy Array) 사이의 직선 거리(L2 Norm)를 반환합니다.
    - 용도: 눈꺼풀 사이의 거리, 입꼬리 사이의 거리 측정 등 기하학적 분석의 기초 함수
    """
    return np.linalg.norm(ptA - ptB)

def get_eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    [EAR(Eye Aspect Ratio) 계산 함수]
    - 목적: 눈이 얼마나 감겨있는지를 수치화하여 졸음을 판단함.
    - 근거 논문: "Real-Time Eye Blink Detection using Facial Landmarks" (Soukupová & Čech, 2016)
    - 원리: 눈이 떠있을 때는 종횡비가 일정 값을 유지하다가, 감을 때 0에 가까워지는 성질을 이용.
    
    Args:
        eye_points: 6개의 (x, y) 좌표를 담은 numpy 배열 (dlib 36~41: 왼쪽 눈, 42~47: 오른쪽 눈)
        
    Returns:
        float: 계산된 EAR 값 (보통 0.2~0.3 이상이면 뜸, 0.2 이하면 감음으로 간주)
    """
    # 1. 수직 눈꺼풀 사이의 거리 (Vertical Distance)
    # 눈의 양쪽 끝을 제외한 위아래 두 쌍의 랜드마크 거리 측정
    # P2(1) <-> P6(5)
    A = euclidean_dist(eye_points[1], eye_points[5])
    # P3(2) <-> P5(4)
    B = euclidean_dist(eye_points[2], eye_points[4])

    # 2. 수평 눈 끝 사이의 거리 (Horizontal Distance)
    # 눈의 양쪽 꼬리(P1, P4) 사이 거리 측정
    # P1(0) <-> P4(3)
    C = euclidean_dist(eye_points[0], eye_points[3])

    # 3. EAR 계산
    # 수직 거리들의 평균 / 수평 거리
    # [안전장치] 분모(C)가 0일 경우를 대비해 아주 작은 값(1e-6)을 더해 ZeroDivisionError 방지
    ear = (A + B) / (2.0 * (C + 1e-6))

    return ear

def get_mouth_width(mouth_points: np.ndarray) -> float:
    """
    [입 가로 너비(Width) 계산]
    - 용도: 'Smile Filter'에서 사용됨.
    - 원리: 하품할 때는 입이 세로로 커지지만, 웃을 때는 가로로 길어짐. 
            이를 구분하기 위해 입꼬리 간의 거리를 측정함.
    
    Args:
        mouth_points: dlib 랜드마크 48~67번 (총 20개 점)
        - Index 0: 48번 (왼쪽 입꼬리)
        - Index 6: 54번 (오른쪽 입꼬리)
    """
    # 48번(Left Corner)과 54번(Right Corner) 사이의 거리 반환
    return euclidean_dist(mouth_points[0], mouth_points[6])

def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    [중요!!] 현재 미사용.. 이 함수를 사용 시 노이즈가 심해져 오히려 인식률이 떨어짐.
            적외선 카메라를 사용하면 이 함수를 사용할 필요가 없음.
    [감마 보정 (Gamma Correction)]
    - 목적: 야간 운전이나 터널 등 조도가 낮은 환경에서 랜드마크 인식률을 높이기 위해 밝기를 비선형적으로 조절함.
    - 특징: 모든 픽셀을 매번 계산하면 느려지므로, Lookup Table(LUT)을 사용하여 연산 속도를 최적화함.
    
    Args:
        image: 입력 이미지 (OpenCV BGR)
        gamma: 감마 값 
               - 1.0: 원본 유지
               - < 1.0: 이미지를 어둡게 (잘 안 씀)
               - > 1.0: 이미지를 밝게 (야간 운전 시 1.5~2.0 권장)
               
    Returns:
        보정된 이미지 (Numpy Array)
    """
    # 감마가 1.0이면 연산 오버헤드를 줄이기 위해 원본 즉시 반환
    if gamma == 1.0:
        return image

    invGamma = 1.0 / gamma
    
    # [최적화] 룩업 테이블(LUT) 생성
    # 0~255 사이의 모든 픽셀 값에 대한 결과값을 미리 계산하여 배열에 저장
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # LUT 적용 (O(1) 매핑으로 고속 처리)
    return cv2.LUT(image, table)