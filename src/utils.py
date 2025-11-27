import numpy as np
import cv2

def euclidean_dist(ptA, ptB):
    # 두 점 사이의 유클리드 거리를 계산합니다.
    return np.linalg.norm(ptA - ptB)

def get_eye_aspect_ratio(eye_points):
    # eye_points는 6개의 (x, y) 좌표를 담은 numpy 배열입니다.

    # 1. 수직 눈꺼풀 사이의 거리를 계산합니다.
    # (P2와 P6 사이, P3와 P5 사이)
    V1 = euclidean_dist(eye_points[1], eye_points[5]) # P2-P6
    V2 = euclidean_dist(eye_points[2], eye_points[4]) # P3-P5

    # 2. 수평 눈 끝 사이의 거리를 계산합니다.
    # (P1과 P4 사이)
    H = euclidean_dist(eye_points[0], eye_points[3]) # P1-P4

    # 3. EAR(눈 종횡비)를 계산합니다.
    ear = (V1 + V2) / (2.0 * H)

    # 4. 계산된 EAR 값을 반환합니다.
    return ear

def adjust_gamma(image, gamma=1.0):
    """
    이미지의 명암(Gamma)을 조절하는 함수
    gamma > 1.0: 이미지가 밝아짐 (어두운 곳/역광 해결)
    gamma < 1.0: 이미지가 어두워짐
    """
    invGamma = 1.0 / gamma
    
    # 룩업 테이블(Lookup Table) 생성 - 속도 최적화
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # 테이블을 이용해 이미지 전체 변환 (매우 빠름)
    return cv2.LUT(image, table)


def get_mouth_width(mouth_points):
    """
    입의 왼쪽 끝(48번)과 오른쪽 끝(54번) 사이의 거리를 계산합니다.
    mouth_points는 dlib 랜드마크 48~67번까지 잘린 배열입니다.
    배열 내부 인덱스: 0번(48번 랜드마크), 6번(54번 랜드마크)
    """
    return euclidean_dist(mouth_points[0], mouth_points[6])

    