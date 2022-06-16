import cv2
import numpy as np


def handle_chamber(gray_frame, bound, center_info, min_thres=195):
    frame_inv = cv2.bitwise_not(gray_frame, mask=gray_frame)
    _, thres = cv2.threshold(frame_inv, min_thres, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, k, iterations=2)
    cnt_closing, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filter_marker = np.zeros(gray_frame.shape, np.uint8)
    for cnt in cnt_closing:
        if cv2.contourArea(cnt) > 3000:
            cv2.drawContours(filter_marker, [cnt], -1, (255, 255, 255), -1)
    filter_marker = cv2.erode(filter_marker, k, iterations=2)

    mask_filter = np.zeros(gray_frame.shape, np.uint8)
    cnt_filter, _ = cv2.findContours(filter_marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnt_filter:
        if cv2.contourArea(c) > 2000:
            cv2.drawContours(mask_filter, [c], -1, (255, 255, 255), -1)
    mask_filter = cv2.erode(mask_filter, k, iterations=2)
    mask_filter[bound != 255] = 0

    # 找出不同腔室位置
    reg_x, reg_y = list(), list()
    cnt_mask, _ = cv2.findContours(mask_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_m_index in range(len(cnt_mask)):
        area = cv2.contourArea(cnt_mask[cnt_m_index])
        if area > 0:
            M = cv2.moments(cnt_mask[cnt_m_index])
            center_chamber_x = int(M["m10"] / M["m00"])
            center_chamber_y = int(M["m01"] / M["m00"])
            reg_x.append(center_chamber_x)
            reg_y.append(center_chamber_y)

        else:
            cv2.drawContours(mask_filter, [cnt_mask[cnt_m_index]], -1, (0, 0, 0), -1)

    # 先用 x 軸抓出 LV 位置後, 再用 y 軸分出 LA 和 Aortic 位置(如果存在的情況)
    # LV 一定在最左邊, 並且小於 整張圖片正中心的位置
    img_center_x, img_center_y = mask_filter.shape
    img_center_x, img_center_y = img_center_x // 2, img_center_y // 2

    reg_sort_x = sorted(reg_x)

    lv_center = list()
    for i in range(len(reg_sort_x)):
        if reg_sort_x[i] < img_center_x:
            lv_pos = reg_x.index(reg_sort_x[i])
            lv_center.append([reg_x[lv_pos], reg_y[lv_pos]])
            del reg_x[lv_pos], reg_y[lv_pos]

    # 若 LV 的中心點大於一個, 則把 LV 給連在一起
    if len(lv_center) > 1:
        reg_px, reg_py = lv_center[0]
        for cen_lv_index in range(0, len(lv_center) - 1):
            p1x, p1y = lv_center[cen_lv_index]
            p2x, p2y = lv_center[cen_lv_index + 1]
            cv2.line(mask_filter, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)
            reg_px += p2x
            reg_py += p2y

        gx, gy = int(reg_px / len(lv_center)), int(reg_py / len(lv_center))
        center_info.extend([[gx, gy]])

    else:
        center_info.extend(lv_center)

    # 拿 muscle B 的 y 軸當成區分 LA 和 Aortic 的標準
    Bx, By = center_info[1]
    la_center, aortic_center = list(), list()
    for j in range(len(reg_y)):
        if reg_y[j] < By:
            la_center.append([reg_x[j], reg_y[j]])

        else:
            aortic_center.append([reg_x[j], reg_y[j]])

    if len(la_center) > 1:
        reg_px, reg_py = la_center[0]
        for cen_la_index in range(0, len(la_center) - 1):
            p1x, p1y = la_center[cen_la_index]
            p2x, p2y = la_center[cen_la_index + 1]
            cv2.line(mask_filter, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)
            reg_px += p2x
            reg_py += p2y
        gx, gy = int(reg_px / len(la_center)), int(reg_py / len(la_center))
        center_info.extend([[gx, gy]])

    else:
        center_info.append(la_center)

    if len(aortic_center) > 1:
        reg_px, reg_py = aortic_center[0]
        for cen_aortic_index in range(0, len(aortic_center) - 1):
            p1x, p1y = aortic_center[cen_aortic_index]
            p2x, p2y = aortic_center[cen_aortic_index + 1]
            cv2.line(mask_filter, (p1x, p1y), (p2x, p2y), (255, 255, 255), 1)
            reg_px += p2x
            reg_py += p2y
        gx, gy = int(reg_px / len(aortic_center)), int(reg_py / len(aortic_center))
        center_info.extend([[gx, gy]])

    else:
        center_info.append(aortic_center)

    return mask_filter, center_info
