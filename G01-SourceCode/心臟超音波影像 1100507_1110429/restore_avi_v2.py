import cv2
import numpy as np
import glob


def read_file(video_dir):
    all_video_path = glob.glob(video_dir + '*.avi')
    return all_video_path


def write_video(last, output_path):
    y, x, _ = last[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))
    for f_index in range(len(last)):
        video_writer.write(last[f_index])
    video_writer.release()


def handle_restore(file_path):
    # 寫入影片用的 list
    last_frames = list()

    video = cv2.VideoCapture(file_path)

    # 避免都卜勒扇形區域破損造成影像缺失
    # _, frame = video.read()
    old_img_contours = np.zeros((600, 800), np.uint8)

    while True:
        ret, frame = video.read()

        if not ret:
            # repeat
            # video = cv2.VideoCapture(case_file_path)
            # continue
            break

        frame_res = frame.copy()
        del_region = frame.copy()

        # 1. 找出扇形 Doppler 區域
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 211]), np.array([180, 30, 255]))

        kernel = np.ones((3, 3), np.uint8)
        mask_white_dilate = cv2.dilate(mask_white, kernel=kernel, iterations=1)

        contours, hier = cv2.findContours(mask_white_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(frame.shape[:2], np.uint8)

        for c in contours:
            area = cv2.contourArea(c)
            cnt_len = cv2.arcLength(c, closed=False)

            if area > 30000 or cnt_len > 500:
                cv2.drawContours(img_contours, [c], -1, (255, 255, 255), -1)

                if np.unique(img_contours, return_counts=True)[1][1] < 30000:
                    img_contours = old_img_contours

                else:
                    old_img_contours = img_contours

        erode = cv2.erode(img_contours, kernel, iterations=2)
        morph_grad = cv2.morphologyEx(erode, cv2.MORPH_GRADIENT, kernel=kernel)  # 原始影像的白色邊框

        # 2. 找出原始影像白色邊框位置, 並且找邊框鄰近區域取平均後取代白色邊框(還原影像)
        # 過濾有顏色的區域
        del_region[erode != 255] = [0, 0, 0]
        del_region[morph_grad == 255] = [0, 0, 0]  # 消除邊框

        del_region_hsv = cv2.cvtColor(del_region, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(del_region_hsv, np.array([0, 43, 46]), np.array([185, 255, 255]))  # 其他的顏色

        # 將有顏色的區域轉成灰階影像
        color_region = frame_res.copy()
        gray_color_region = cv2.cvtColor(color_region, cv2.COLOR_BGR2GRAY)
        gray_color_region[mask_color != 255] = 0

        scale = int(np.max(gray_color_region) * 0.14)
        gray_color_region = np.clip(gray_color_region, 0, scale, out=gray_color_region)
        bgr_color_region = cv2.cvtColor(gray_color_region, cv2.COLOR_GRAY2BGR)

        frame_res[mask_color == 255] = [0, 0, 0]

        # 消除白色邊框(用中值濾波)
        # 取中值濾波
        gray = cv2.cvtColor(frame_res, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        blur[morph_grad != 255] = 0

        blur_bgr = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        # 兩影像疊加
        frame_res[morph_grad == 255] = [0, 0, 0]
        frame_res = cv2.addWeighted(frame_res, 1, blur_bgr, 0.9, 1)
        frame_res = cv2.addWeighted(frame_res, 1, bgr_color_region, 1, 1)
        last_frames.append(frame_res)
        # cv2.imshow('frame_res', frame_res)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        elif key == ord('p'):
            while cv2.waitKey(1) != ord(' '):
                pass

    return last_frames


if __name__ == '__main__':
    path = './video/1st data all/'
    all_avi_files = read_file(path)

    output_dir = './video/1st data all restore/'
    for path in all_avi_files:
        file_name = path.split('\\')[-1]
        result = handle_restore(path)
        output_path = output_dir + file_name
        write_video(result, output_path)
