import cv2
import numpy as np


def muscle_bounding(src, min_thres=50, erode_iterations=10):
    """
    function:
        muscle_bounding(src[, min_thres=50[, erode_iterations=10]]):
            找出肌肉的邊界

    parameter:
        src: matching 合併後的圖像
        min_thres: 二值化的最小門檻值
        erode_iterations: 侵蝕次數

    return:
        erode_bound_region: 侵蝕過後的肌肉邊界區域(目的用來抓腔室位置)
    """
    _, thres = cv2.threshold(src, min_thres, 255, cv2.THRESH_BINARY)
    gray_muscle = cv2.cvtColor(thres, cv2.COLOR_BGR2GRAY)
    cnt_muscle, _ = cv2.findContours(gray_muscle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_muscle = list()
    for i in range(len(cnt_muscle)):
        for j in range(len(cnt_muscle[i])):
            hull_muscle.append(cnt_muscle[i][j])

    hull_muscle = np.asarray(hull_muscle)
    hull_muscle = cv2.convexHull(hull_muscle)

    bound_region = np.zeros(src.shape, np.uint8)
    bound_line = np.zeros(src.shape, np.uint8)
    cv2.drawContours(bound_region, [hull_muscle], 0, (255, 255, 255), -1)
    cv2.drawContours(bound_line, [hull_muscle], 0, (255, 255, 255), 2)

    gray_bound_region = cv2.cvtColor(bound_region, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode_bound_region = cv2.erode(gray_bound_region, k, iterations=erode_iterations)

    return gray_bound_region, erode_bound_region
