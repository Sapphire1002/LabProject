import cv2
import numpy as np


def handle_watershed(frame, bound, mark, center_info):
    bg = cv2.dilate(bound, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    unknown = cv2.subtract(bg, mark)

    _, markers = cv2.connectedComponents(mark)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)
    uni_mark, count = np.unique(markers, return_counts=True)

    color_info = np.zeros(frame.shape, np.uint8)
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255]
    ]

    reg_pos = np.zeros(frame.shape[:2], np.uint8)
    bound_info = np.zeros(frame.shape, np.uint8)

    for m in range(len(uni_mark)):
        index, min_dis = 0, 600

        # 過濾太小區域
        if uni_mark[m] > 1 and count[m] > 20:
            reg_pos[markers == uni_mark[m]] = 255

            cnt_reg, _ = cv2.findContours(reg_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            M = cv2.moments(cnt_reg[0])
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # 計算 center_x, center_y 和 肌肉 & 腔室位置的最短距離
            for d in range(len(center_info)):
                if len(center_info[d]) != 0:
                    if isinstance(center_info[d][0], list):
                        for i in range(len(center_info[d])):
                            xx, yy = center_info[d][i]
                            distance = np.sqrt((center_x - xx) ** 2 + (center_y - yy) ** 2)
                            if min_dis > distance:
                                min_dis = distance
                                index = d
                    else:
                        xx, yy = center_info[d]
                        distance = np.sqrt((center_x - xx) ** 2 + (center_y - yy) ** 2)
                        if min_dis > distance:
                            min_dis = distance
                            index = d

            color_info[markers == uni_mark[m]] = colors[index]
            b, g, r = colors[index]
            cv2.drawContours(bound_info, cnt_reg, -1, (b, g, r), 2)
            reg_pos[markers == uni_mark[m]] = 0

    curr_result = cv2.addWeighted(frame, 1, color_info, 0.1, 1)
    curr_result = cv2.addWeighted(curr_result, 1, bound_info, 1, 1)

    return curr_result
