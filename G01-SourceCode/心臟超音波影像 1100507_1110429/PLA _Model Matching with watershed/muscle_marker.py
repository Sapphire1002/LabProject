import cv2
import numpy as np


def conv_gray(src):
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


def conv_thres(src, min_thres):
    return cv2.threshold(src, min_thres, 255, cv2.THRESH_BINARY)[1]


def find_contours(src):
    return cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def handle_valve(cntA, cntB, cntC):
    all_bx, all_by = list(), list()
    for cntB_index in range(len(cntB)):
        for B_point_index in range(len(cntB[cntB_index])):
            bx, by = cntB[cntB_index][B_point_index][0]
            all_bx.append(bx)
            all_by.append(by)

    all_A_point = list()
    for cntA_index in range(len(cntA)):
        for A_point_index in range(len(cntA[cntA_index])):
            ax, ay = cntA[cntA_index][A_point_index][0]
            all_A_point.append([ax, ay])

    all_C_point = list()
    for cntC_index in range(len(cntC)):
        for C_point_index in range(len(cntC[cntC_index])):
            cx, cy = cntC[cntC_index][C_point_index][0]
            all_C_point.append([cx, cy])

    # mitral valve search range
    search_range = (max(all_bx) - min(all_bx)) * 0.2 + min(all_bx)
    mv_range = list()
    for f in range(len(all_bx)):
        if all_bx[f] <= search_range:
            mv_range.append([all_bx[f], all_by[f]])

    min_dis = 600
    mv_ax, mv_ay = 0, 0
    mv_bx, mv_by = 0, 0
    for a_index in range(len(all_A_point)):
        ax, ay = all_A_point[a_index]
        for b_index in range(len(mv_range)):
            bx, by = mv_range[b_index]
            distance = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

            if distance < min_dis:
                mv_ax, mv_ay = ax, ay
                mv_bx, mv_by = bx, by
                min_dis = distance

    # aortic valve search range
    max_search_range = (max(all_bx) - min(all_bx)) * 0.6 + min(all_bx)
    min_search_range = (max(all_bx) - min(all_bx)) * 0.3 + min(all_bx)
    av_range = list()
    for f in range(len(all_bx)):
        if max_search_range > all_bx[f] >= min_search_range:
            av_range.append([all_bx[f], all_by[f]])

    min_dis = 600
    av_cx, av_cy = 0, 0
    av_bx, av_by = 0, 0
    for c_index in range(len(all_C_point)):
        cx, cy = all_C_point[c_index]
        for b_index in range(len(av_range)):
            bx, by = av_range[b_index]
            distance = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

            if distance < min_dis:
                av_cx, av_cy = cx, cy
                av_bx, av_by = bx, by
                min_dis = distance

    return (mv_ax-30, mv_ay-10), (mv_bx-30, mv_by+10), (av_cx-25, av_cy-10), (av_bx-25, av_by+10)


def handle_muscle(muscles, muscleA, muscleB, muscleC, min_thres=50):
    mask_valve = np.zeros(muscles.shape[:2], np.uint8)
    muscle_center_info = list()

    gray_A = conv_gray(muscleA)
    gray_B = conv_gray(muscleB)
    gray_C = conv_gray(muscleC)

    thresA = conv_thres(gray_A, min_thres)
    thresB = conv_thres(gray_B, min_thres)
    thresC = conv_thres(gray_C, min_thres)

    cntA = find_contours(thresA)
    cntB = find_contours(thresB)
    cntC = find_contours(thresC)

    mv_pt1, mv_pt2, av_pt1, av_pt2 = handle_valve(cntA, cntB, cntC)
    cv2.line(mask_valve, mv_pt1, mv_pt2, (255, 255, 255), 20)
    cv2.line(mask_valve, av_pt1, av_pt2, (255, 255, 255), 20)
    # valve end.

    # muscle
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erodeA = cv2.erode(thresA, k, iterations=5)
    erodeB = cv2.erode(thresB, k, iterations=5)
    erodeC = cv2.erode(thresC, k, iterations=5)

    cntA = find_contours(erodeA)
    cntB = find_contours(erodeB)
    cntC = find_contours(erodeC)

    # 找出 erode 後的所有輪廓中心點
    all_centerA = list()
    filter_area = 20
    for cntA_index in range(len(cntA)):
        area = cv2.contourArea(cntA[cntA_index])

        if area > filter_area:
            M = cv2.moments(cntA[cntA_index])
            center_ax = int(M["m10"] / M["m00"])
            center_ay = int(M["m01"] / M["m00"])
            all_centerA.append([center_ax, center_ay])

        else:
            cv2.drawContours(erodeA, [cntA[cntA_index]], -1, (0, 0, 0), -1)

    all_centerB = list()
    for cntB_index in range(len(cntB)):
        area = cv2.contourArea(cntB[cntB_index])

        if area > filter_area:
            M = cv2.moments(cntB[cntB_index])
            center_bx = int(M["m10"] / M["m00"])
            center_by = int(M["m01"] / M["m00"])
            all_centerB.append([center_bx, center_by])

        else:
            cv2.drawContours(erodeB, [cntB[cntB_index]], -1, (0, 0, 0), -1)

    all_centerC = list()
    for cntC_index in range(len(cntC)):
        area = cv2.contourArea(cntC[cntC_index])

        if area > filter_area:
            M = cv2.moments(cntC[cntC_index])
            center_cx = int(M["m10"] / M["m00"])
            center_cy = int(M["m01"] / M["m00"])
            all_centerC.append([center_cx, center_cy])

        else:
            cv2.drawContours(erodeC, [cntC[cntC_index]], -1, (0, 0, 0), -1)

    # 連接所有中心點
    for cen_A_index in range(0, len(all_centerA) - 1):
        pt1x, pt1y = all_centerA[cen_A_index]
        pt2x, pt2y = all_centerA[cen_A_index + 1]
        cv2.line(erodeA, (pt1x, pt1y), (pt2x, pt2y), (255, 255, 255), 1)

    for cen_B_index in range(0, len(all_centerB) - 1):
        pt1x, pt1y = all_centerB[cen_B_index]
        pt2x, pt2y = all_centerB[cen_B_index + 1]
        cv2.line(erodeB, (pt1x, pt1y), (pt2x, pt2y), (255, 255, 255), 1)

    for cen_C_index in range(0, len(all_centerC) - 1):
        pt1x, pt1y = all_centerC[cen_C_index]
        pt2x, pt2y = all_centerC[cen_C_index + 1]
        cv2.line(erodeC, (pt1x, pt1y), (pt2x, pt2y), (255, 255, 255), 1)

    # 再找一次輪廓, 最後找出屬於該區段 marker 的正中心點
    cntA = find_contours(erodeA)
    cntB = find_contours(erodeB)
    cntC = find_contours(erodeC)

    M = cv2.moments(cntA[0])
    center_ax = int(M["m10"] / M["m00"])
    center_ay = int(M["m01"] / M["m00"])
    muscle_center_info.extend([[center_ax, center_ay]])

    M = cv2.moments(cntB[0])
    center_bx = int(M["m10"] / M["m00"])
    center_by = int(M["m01"] / M["m00"])
    muscle_center_info.extend([[center_bx, center_by]])

    M = cv2.moments(cntC[0])
    center_cx = int(M["m10"] / M["m00"])
    center_cy = int(M["m01"] / M["m00"])
    muscle_center_info.extend([[center_cx, center_cy]])

    mask_muscle = cv2.add(erodeA, cv2.add(erodeB, erodeC))

    return mask_valve, mask_muscle, muscle_center_info
