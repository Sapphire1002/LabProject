import numpy as np
import cv2


class FindROI(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

        if not self.video:
            raise Exception('影片檔案路徑不存在或格式錯誤')

        else:
            self.ret = True
            self.roi = None
            self.ox, self.oy = 0, 0
            self.radius = 0

    def get_frame(self):
        self.ret, curr_frame = self.video.read()
        return curr_frame

    def roi_region(self, input_path, mask_threshold=10, kernel_size=(3, 3), morph_iter=3):
        handle_video = cv2.VideoCapture(input_path)
        _, first = handle_video.read()
        mask_diff_all = np.zeros(first.shape[:2], np.uint8)

        while True:
            _ret, f = handle_video.read()

            if not _ret:
                break

            gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_frame, gray_first)

            mask_diff = np.zeros(diff.shape, np.uint8)
            mask_diff[diff > mask_threshold] = 255
            mask_diff_all += mask_diff
            np.clip(mask_diff_all, 0, 255, out=mask_diff_all)

        mask_last = np.zeros(first.shape[:2], np.uint8)
        contours, hier = cv2.findContours(mask_diff_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_last, contours, -1, (255, 255, 255), -1)

        kernel = np.ones(kernel_size, np.uint8)
        mask_erode = cv2.erode(mask_last, kernel, iterations=morph_iter)
        mask_dilate = cv2.dilate(mask_erode, kernel, iterations=morph_iter - 1)

        mask_last_bound = np.zeros(first.shape[:2], np.uint8)
        contours, hier = cv2.findContours(mask_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 300:
                cv2.drawContours(mask_last_bound, [c], -1, (255, 255, 255), 2)
        contours, hier = cv2.findContours(mask_last_bound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 已知扇形角度為 90, 直線斜率為 1, -1
        # 霍夫轉換找出直線後, 找交點求圓心
        roi_pos = np.zeros(mask_dilate.shape, np.uint8)

        lines = cv2.HoughLinesP(mask_last_bound, 1, np.pi / 180, threshold=200, minLineLength=60, maxLineGap=130)
        x1_y1 = list()
        x2_y2 = list()
        lm_error, rm_error = 1, 1
        l_index = None
        r_index = None

        try:
            # 找出線的斜率後判斷為左邊或右邊
            for line_index in range(len(lines)):
                x1, y1, x2, y2 = lines[line_index][0]
                m = (y2 - y1) / ((x2 - x1) + 1e-08)

                if m < 0:
                    if abs(m + 1) < lm_error:
                        lm_error = abs(m + 1)
                        l_index = line_index
                else:
                    if abs(m - 1) < rm_error:
                        rm_error = abs(m - 1)
                        r_index = line_index

                x1_y1.append((x1, y1))
                x2_y2.append((x2, y2))

            # 找到最接近斜率 = -1, 1 的線段, 並找出圓心座標
            a1, b1 = x1_y1[l_index]
            a2, b2 = x2_y2[l_index]
            m1 = (b2 - b1) / (a2 - a1)

            A1, B1 = x1_y1[r_index]
            A2, B2 = x2_y2[r_index]
            m2 = (B2 - B1) / (A2 - A1)
            c0, c1 = m1 * a1 - b1, m2 * A1 - B1

            ox = np.round((c0 - c1) / (m1 - m2)).astype(np.int)
            oy = np.round(((m1 + m2) * ox - c0 - c1) / 2).astype(np.int)

            rad = 0
            for ii in range(len(contours)):
                for jj in range(len(contours[ii])):
                    if rad < contours[ii][jj][0][1]:
                        rad = contours[ii][jj][0][1]
            rad = rad - oy
            cv2.ellipse(roi_pos, (ox, oy), (rad, rad), 90, -45, 45, (255, 255, 255), -1)

        except TypeError:
            # 假如找不到直線就以最低點當成圓心, 畫出 90 度的扇形, 半徑為 y 軸的最高減圓心
            rad = 0
            ox, oy = 0, 600

            for i in range(len(contours)):
                for j in range(len(contours[i])):
                    if rad < contours[i][j][0][1]:
                        rad = contours[i][j][0][1]

                    if oy > contours[i][j][0][1]:
                        oy = contours[i][j][0][1]
                        ox = contours[i][j][0][0]
            rad = rad - oy
            cv2.ellipse(roi_pos, (ox, oy), (rad, rad), 90, -45, 45, (255, 255, 255), -1)

        self.roi = roi_pos
        self.ox = ox
        self.oy = oy
        self.radius = rad

# using
# video = VideoInit(filepath)
# video.roi_region(filepath)
# mask_roi = video.roi
# frame[mask_roi != 255] = [0, 0, 0]
