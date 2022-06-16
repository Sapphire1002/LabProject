from find_roi import FindROI
from skeletonize_bound import skeleton_bound
from model_scale import adjust_scale
from matching import match_muscleA, match_muscleB, match_muscleC
from muscle_bound import muscle_bounding
from muscle_marker import handle_muscle
from chamber_marker import handle_chamber
from mark_position import handle_watershed

import glob
import cv2
import numpy as np
import time


def read_file(video_dir):
    all_video_path = glob.glob(video_dir + '*.avi')
    return all_video_path


def write_video(frames_list, output_path):
    y, x, _ = frames_list[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))
    for i in frames_list:
        video_writer.write(i)
    video_writer.release()


video_path = read_file('..\\2nd data class 9\\')
skeletonize_dir = '..\\2nd data class 9 skeletonize\\'

model_all = cv2.imread('../model_anchor/0009_Parasternal long axis.png')
model_A = cv2.imread('../model_anchor/0009_Parasternal long axis_1.png')
model_B = cv2.imread('../model_anchor/0009_Parasternal long axis_2.png')
model_C = cv2.imread('../model_anchor/0009_Parasternal long axis_3.png')

for path in video_path:
    # pre_st_time = time.time()
    print(path)
    # 檔案名稱
    file_name = path.split('\\')[-1]

    # 找出影片的 ROI
    video = FindROI(path)
    video.roi_region(path)
    mask_roi = video.roi

    # 處理骨架圖片
    skeletonize_file_path = skeletonize_dir + file_name + '.png'

    # 取得骨架邊界(目的調整 ROI 的範圍)
    skeleton_info = skeleton_bound(skeletonize_file_path, mask_roi)
    if skeleton_info is None:
        continue

    top, bottom, left, right, radius = skeleton_info
    mask_roi = mask_roi[top:bottom, left:right]

    # 調整標準模型的大小比例
    # 用 Parasternal long axis 已知條件來寫
    output_model_A, output_model_B, output_model_C = adjust_scale(model_all, model_A, model_B, model_C, radius)

    # pre_end_time = time.time()
    # print('前處理的時間:', round(pre_end_time - pre_st_time, 3), '秒')

    # 根據每幀抓取肌肉位置
    video = cv2.VideoCapture(path)
    curr_res = list()

    # video_st_time = time.time()
    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = frame[top:bottom, left:right]
        frame[mask_roi != 255] = [0, 0, 0]
        frame_cp = frame.copy()

        # matching 順序 A -> C -> B
        # matching_st_time = time.time()
        model_best_A, best_fitting_A = match_muscleA(frame, output_model_A)

        # 將 A 區塊消除
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        Area_A = cv2.dilate(best_fitting_A, k, iterations=15)
        match_seg_C = frame.copy()
        match_seg_C[Area_A == 255] = 0

        model_best_C, best_fitting_C = match_muscleC(frame, output_model_C, seg_C=match_seg_C)

        # 將 A, C 區塊消除
        Area_C = cv2.dilate(best_fitting_C, k, iterations=10)
        match_seg_B = frame.copy()
        match_seg_B[Area_A == 255] = 0
        match_seg_B[Area_C == 255] = 0

        model_best_B, best_fitting_B = match_muscleB(frame, output_model_B, seg_B=match_seg_B)
        # matching_end_time = time.time()
        # print('matching 肌肉時間:', round(matching_end_time - matching_st_time, 3), '秒')

        # 將 A, B, C 的結果合併
        result_A = cv2.addWeighted(best_fitting_A, 0.7, model_best_A, 0.3, 1)
        result_B = cv2.addWeighted(best_fitting_B, 0.7, model_best_B, 0.3, 1)
        result_C = cv2.addWeighted(best_fitting_C, 0.7, model_best_C, 0.3, 1)

        muscle_all_fitting = cv2.add(cv2.add(result_A, result_B), result_C)
        muscle_region = best_fitting_A + best_fitting_B + best_fitting_C

        # 找肌肉邊界
        ori_bound, erode_bound = muscle_bounding(muscle_all_fitting, 50, 10)

        # 找出肌肉 marker
        mask_valve, mask_muscle, muscle_center_info = handle_muscle(
            muscle_region,
            best_fitting_A,
            best_fitting_B,
            best_fitting_C,
            50
        )
        # cv2.imshow('mask_valve', mask_valve)
        # cv2.imshow('mask_muscle', mask_muscle)

        # 找出腔室 marker
        frame_cp[erode_bound != 255] = [0, 0, 0]
        gray_frame = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2GRAY)
        gray_frame[mask_valve == 255] = 255
        mask_filter, center_info = handle_chamber(gray_frame, erode_bound, muscle_center_info, 195)
        mask_filter[mask_muscle == 255] = 255
        cv2.imshow('mask_filter_res', mask_filter)
        # print('center info:', center_info)

        # watershed
        curr_result = handle_watershed(frame, ori_bound, mask_filter, center_info)
        curr_res.append(curr_result)
        cv2.imshow('curr_result', curr_result)
        cv2.waitKey(1)

    write_path = '../2nd data class 9 result/' + file_name
    print(write_path)
    write_video(curr_res, write_path)

    # video_end_time = time.time()
    # print('%s 影片處理的時間:' % file_name, round(video_end_time - video_st_time, 3), '秒')
