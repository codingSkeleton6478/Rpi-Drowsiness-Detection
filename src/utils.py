import numpy as np

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