import cv2
import numpy as np
import pydicom
import time


class DopplerModel(object):
    def __init__(self, video_path, diagnosisPos, case_name):
        self.video = cv2.VideoCapture(video_path)
        self.path = video_path

        # 診斷的位置
        self.diag_pos = diagnosisPos
        # 病歷號碼
        self.case_name = case_name

        if not self.video.isOpened():
            raise Exception('影片檔案路徑不存在或格式錯誤')

        else:
            # 影片初始化資訊
            self.ret = True
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # get_frame 屬性
        self.count_list = list()
        self.curr_frame_count = 0

        # __roi 的屬性
        self.roi = None
        self.ox, self.oy = 0, 0
        self.radius = 0

        # standard_uint 屬性
        self.scale = 0

        # color_info 屬性
        self.all_center = None
        self.connect_index = None
        self.connect_area = None
        self.HRange = None
        self.effect_area = list()
        self.que_frame = list()

        # handle degree 的四分位差的結果寫成定值
        self.aortic_Q25, self.aortic_Q50, self.aortic_Q75 = [0.7775, 1.445, 2.3025]
        self.mitral_Q25, self.mitral_Q50, self.mitral_Q75 = [1.145, 2.22, 3.375]
        self.pulmonary_Q25, self.pulmonary_Q50, self.pulmonary_Q75 = [0.3975, 0.795, 1.5075]
        self.tricuspid_Q25, self.tricuspid_Q50, self.tricuspid_Q75 = [0.6275, 1.495, 2.605]
        self.level = ''
        self.degree_name = ['Severe', 'Moderate', 'Mild', 'Trivial']

        # show_TimeBar 屬性
        self.que_degree = list()

    def get_frame(self):
        """
        function:
            get_frame(self): 讀取每一幀

        return:
            curr_frame: 當前 frame
        """
        self.ret, curr_frame = self.video.read()
        self.curr_frame_count = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        return curr_frame

    def __roi(self):
        """
        function:
            __roi(self): 找出影片的 ROI 有效區域

        method:
            1. 用第一幀和每幀的差異疊加 mask
            2. 將疊加後的 mask, 找出 contour 後繪製實心輪廓
            3. 將實心輪廓 做形態學(先侵蝕後膨脹)
            4. 畫出實心輪廓的邊界(mask_last_bound)
            5. 霍夫轉換找 mask_last_bound 的直線
            6-1. 找出線的斜率, 兩線的交點為圓心畫 90 度扇形
            6-2. 若找不到直線, 找出 mask_last_bound 的最高當圓心, 最高和對低點當半徑畫 90 度扇形

        return:
            roi_pos: 影片的 ROI 區域, ndarray, ndim=2
            ox: 圓心的 x 座標
            oy: 圓心的 y 座標
            radius: 圓半徑
        """
        # 1. 用第一幀和每幀的差異疊加 mask
        video = cv2.VideoCapture(self.path)
        _, first = video.read()
        mask_diff_all = np.zeros(first.shape[:2], np.uint8)

        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_frame, gray_first)

            mask_diff = np.zeros(diff.shape, np.uint8)
            mask_diff[diff > 10] = 255
            mask_diff_all += mask_diff
            np.clip(mask_diff_all, 0, 255, out=mask_diff_all)

        # 2. 將疊加後的 mask, 找出 contour 後繪製實心輪廓
        mask_last = np.zeros(first.shape[:2], np.uint8)
        cnt_mask_diff_all, _ = cv2.findContours(mask_diff_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_last, cnt_mask_diff_all, -1, (255, 255, 255), -1)

        # 3. 將實心輪廓 做形態學(先侵蝕後膨脹)
        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(mask_last, kernel, iterations=3)
        dilate = cv2.dilate(erode, kernel, iterations=2)

        # 4. 畫出實心輪廓的邊界(mask_last_bound)
        mask_last_bound = np.zeros(first.shape[:2], np.uint8)
        cnt_bound, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_last_bound, cnt_bound, -1, (255, 255, 255), 2)

        # 5. 霍夫轉換找 mask_last_bound 的直線
        roi_pos = np.zeros(dilate.shape, np.uint8)

        lines = cv2.HoughLinesP(
            mask_last_bound,
            1,
            np.pi / 180,
            threshold=200,
            minLineLength=60,
            maxLineGap=130
        )

        # 6-1. 找出線的斜率, 兩線的交點為圓心畫 90 度扇形
        x1_y1, x2_y2 = list(), list()
        lm_error, rm_error = 1, 1
        l_index, r_index = None, None

        try:
            for line_index in range(len(lines)):
                x1, y1, x2, y2 = lines[line_index][0]
                m = (y2 - y1) / ((x2 - x1) + 1e-08)

                if m < 0:
                    if abs(m + 1) < lm_error:
                        rm_error = abs(m + 1)
                        l_index = line_index
                else:
                    if abs(m - 1) < rm_error:
                        rm_error = abs(m - 1)
                        r_index = line_index

                x1_y1.append((x1, y1))
                x2_y2.append((x2, y2))

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
            for i in range(len(cnt_bound)):
                for j in range(len(cnt_bound[i])):
                    if rad < cnt_bound[i][j][0][1]:
                        rad = cnt_bound[i][j][0][1]
            rad -= oy
            cv2.ellipse(roi_pos, (ox, oy), (rad, rad), 90, -45, 45, (255, 255, 255), -1)

        except TypeError:
            # 6-2. 若找不到直線, 找出 mask_last_bound 的最高當圓心, 最高和對低點當半徑畫 90 度扇形
            rad = 0
            ox, oy = 0, 600

            for i in range(len(cnt_bound)):
                for j in range(len(cnt_bound[i])):
                    if rad < cnt_bound[i][j][0][1]:
                        rad = cnt_bound[i][j][0][1]

                    if oy > cnt_bound[i][j][0][1]:
                        oy = cnt_bound[i][j][0][1]
                        ox = cnt_bound[i][j][0][0]
            rad = rad - oy
            cv2.ellipse(roi_pos, (ox, oy), (rad, rad), 90, -45, 45, (255, 255, 255), -1)

        self.roi = roi_pos
        self.ox = ox
        self.oy = oy
        self.radius = rad

    def standard_unit(self, unit_len):
        """
        function:
            standard_uint(self, scale): 找出影像的標準單位(pixel to cm)

        parameter:
            unit_len: 當前影像的標準單位長度 cm, int. (目前以手動輸入方式處理)
            (自動處理要嘗試找出標準單位長度的文字)
        """
        self.__roi()
        self.scale = self.radius / unit_len

    def find_region(self):
        """
        function:
            find_region(self): 找出 Color Doppler 有效扇形區域

        return:
            mask_region: 傳回 Color Doppler 有效區域遮罩, 二值圖
        """
        video = cv2.VideoCapture(self.path)

        while True:
            r, frame = video.read()

            if not r:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_white = cv2.inRange(hsv, np.array([0, 0, 211]), np.array([180, 30, 255]))

            k = np.ones((3, 3), np.uint8)
            mask_white_dilate = cv2.dilate(mask_white, k, iterations=1)

            cnt, _ = cv2.findContours(mask_white_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_region = np.zeros((self.height, self.width), np.uint8)

            for c in cnt:
                area = cv2.contourArea(c)
                cnt_len = cv2.arcLength(c, closed=False)

                # 過濾白色文字
                if area > 40000 or cnt_len > 500:
                    cv2.drawContours(mask_region, [c], -1, (255, 255, 255), -1)

                    # 判斷扇形是否缺陷
                    if np.unique(mask_region, return_counts=True)[1][1] > 30000:
                        mask_region = cv2.erode(mask_region, k, iterations=2)
                        return mask_region

                    # 直到找到符合的為止
                    else:
                        continue

    def color_info(self, frame, HueRange=(26, 99), maxDis=60):
        """
        function:
            color_info(frame[, HueRange=(26, 99)[, maxDis=60]]): 找出有問題的顏色區域

        parameter:
            frame: 影片的每一幀(經遮罩後)
            HueRange: HSV 色相(H)的範圍, 紅(H=0), 藍(H=124). 目前採用 H=26, 99, int
            maxDis: 有問題區域輪廓的距離門檻值, 默認 60, int

        method:
            1. HSV 找出有問題的區域後, 開運算過濾雜點得到 mask_others
            2. 找出輪廓, 計算有問題區域的面積
            3. 將鄰近輪廓連接在一起(輪廓間的最短距離), 以及連接起來的面積
            4. 接著再過濾掉過小區域的部分

        return:
            mask_connect: 傳回連接 connect 後的遮罩, 二值圖
        """
        # 1.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.HRange = HueRange
        mask_others = cv2.inRange(hsv, np.array([HueRange[0], 43, 46]), np.array([HueRange[1], 255, 255]))

        k = np.ones((3, 3), np.uint8)
        mask_others = cv2.morphologyEx(mask_others, cv2.MORPH_OPEN, k)

        # 2.
        cnt_others, _ = cv2.findContours(mask_others, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_area = list()
        for cnt in cnt_others:
            area = cv2.contourArea(cnt)
            all_area.append(area)

        if len(all_area) == 0:
            return None

        # 3.
        mask_connect = mask_others.copy()

        # 利用 cv2.moments 找出輪廓的中心點
        all_center = list()
        for cnt in cnt_others:
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            all_center.append((cx, cy))

        connect_index = list()
        is_connected = np.ones(len(cnt_others))

        for cnt_init_index in range(len(all_center)):
            if not is_connected[cnt_init_index]:
                continue

            x_init, y_init = all_center[cnt_init_index]
            curr_index = list()
            curr_next_conn = 0

            if len(connect_index) > 0:
                if len(connect_index[-1]) == 1:
                    curr_index.append(cnt_init_index)

                else:
                    for curr_init in range(0, len(all_center)):
                        if curr_init not in connect_index[-1]:
                            continue

                        else:
                            conn_curr_init = connect_index[-1].index(curr_init)

                        x_init2, y_init2 = all_center[connect_index[-1][conn_curr_init]]

                        for curr_next in range(0, len(all_center)):
                            if is_connected[curr_next]:
                                x_end2, y_end2 = all_center[curr_next]
                                dis1 = np.abs(cv2.pointPolygonTest(cnt_others[curr_init], (x_end2, y_end2), True))
                                dis2 = np.abs(cv2.pointPolygonTest(cnt_others[curr_next], (x_init2, y_init2), True))

                                if dis2 <= maxDis or dis1 <= maxDis:
                                    connect_index[-1].append(curr_next)
                                    is_connected[curr_next] = 0
                                    x1, y1, x2, y2 = int(x_init2), int(y_init2), int(x_end2), int(y_end2)

                                    # 若連接的中心點為0, 使用輪廓第一個點相連
                                    if mask_connect[y1, x1] == 0 or mask_connect[y2, x2] == 0:
                                        x1, y1 = cnt_others[curr_init][0][0]
                                        x2, y2 = cnt_others[curr_next][0][0]
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                                        cv2.line(mask_connect, (x1, y1), (x2, y2), (255, 255, 255), 1)

                                    else:
                                        cv2.line(mask_connect, (x1, y1), (x2, y2), (255, 255, 255), 1)
                                else:
                                    curr_next_conn = 1

            if curr_next_conn:
                check = 1
                for is_connect_index in range(len(connect_index)):
                    if cnt_init_index in connect_index[is_connect_index]:
                        curr_index = connect_index[is_connect_index]
                        check = 0

                    elif is_connect_index == len(connect_index) - 1 and check:
                        curr_index.append(cnt_init_index)

            if len(connect_index) == 0:
                curr_index.append(cnt_init_index)
            is_connected[cnt_init_index] = 0

            for cnt_next_index in range(cnt_init_index + 1, len(all_center)):
                if is_connected[cnt_next_index]:
                    x_end, y_end = all_center[cnt_next_index]
                    dis1 = np.abs(cv2.pointPolygonTest(cnt_others[cnt_init_index], (x_end, y_end), True))
                    dis2 = np.abs(cv2.pointPolygonTest(cnt_others[cnt_next_index], (x_init, y_init), True))

                    # 判斷兩點距離
                    if dis1 <= maxDis or dis2 <= maxDis:
                        # 若 next node 沒有被連過, 就加到 curr_index
                        if cnt_next_index not in curr_index and is_connected[cnt_next_index]:
                            curr_index.append(cnt_next_index)
                            is_connected[cnt_next_index] = 0

                            # contour connect
                            x_init, y_init, x_end, y_end = int(x_init), int(y_init), int(x_end), int(y_end)

                            # 若連接的中心點為0, 使用輪廓第一個點相連
                            if mask_connect[y_end, x_end] == 0 or mask_connect[y_init, x_init] == 0:
                                x_init, y_init = cnt_others[cnt_init_index][0][0]
                                x_end, y_end = cnt_others[cnt_next_index][0][0]
                                x_init, y_init, x_end, y_end = int(x_init), int(y_init), int(x_end), int(y_end)

                                cv2.line(mask_connect, (x_init, y_init), (x_end, y_end), (255, 255, 255), 1)

                            else:
                                cv2.line(mask_connect, (x_init, y_init), (x_end, y_end), (255, 255, 255), 1)

            # 避免加入重複的元素和空列表
            if curr_index not in connect_index and len(curr_index) > 0:
                connect_index.append(curr_index)

        # 計算連接輪廓後的面積
        connect_area = list()
        for count in connect_index:
            area = 0
            for index in count:
                area += all_area[index]
            connect_area.append(area)

        self.all_center = all_center
        self.connect_index = connect_index
        self.connect_area = connect_area
        return mask_connect

    def show(self, frame, mask_connect, minArea=0.1, personalInfo=False, dcm_path=None):
        """
        function:
            show(frame, mask_connect[, minArea=0.1[, personalInfo]]): 展示 Color Doppler 結果

        parameter:
            frame: 原始影像的每一幀
            mask_connect: 連接 contours 後的而值化結果
            minArea: 有問題區域的最小面積, float, 默認 0.1 平方公分
            personalInfo: 是否顯示個人資訊
            dcm_path: DCM 檔案路徑, str

        return:
            curr_res: 當前呈現的結果
        """
        draw_img = np.zeros(frame.shape, np.uint8)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        data_info = list()

        if mask_connect is not None:
            cnt_connect, _ = cv2.findContours(mask_connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # -- debug 用
            # if len(cnt_connect) != len(self.connect_area):
            #     print(len(cnt_connect), len(self.connect_area))
            #     cv2.imshow('mask conn', mask_connect)
            #     cv2.waitKey(0)
            # # --

            # 找出相對應面積的長寬
            for cnt_conn_index in range(len(cnt_connect)):
                # 判斷是否實際為逆流, 或者只是單純數值高而已
                check_regurgitation = np.zeros(frame.shape[:2], np.uint8)

                if len(cnt_connect) == 1:
                    # curr: 用最小擬合矩形框出
                    rect = cv2.minAreaRect(cnt_connect[cnt_conn_index])
                    points = cv2.boxPoints(rect).astype(np.int0)

                    # 判斷該區域是否有逆流
                    cv2.drawContours(check_regurgitation, [points], -1, (255, 255, 255), -1)
                    check_frame = frame.copy()
                    check_frame[check_regurgitation != 255] = [0, 0, 0]
                    check_hsv = cv2.cvtColor(check_frame, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(check_hsv)
                    h_min, h_max = self.HRange
                    h[h < h_min] = 0
                    h[h > h_max] = 0

                    uni_h, count = np.unique(h, return_counts=True)
                    uni_h, count = uni_h[1:], count[1:]  # 去除 0 的部分
                    blue_count = np.sum(count[uni_h > 77])
                    red_count = np.sum(count[uni_h <= 35])

                    regurgitation_scale = np.minimum(red_count, blue_count) / (np.maximum(red_count, blue_count) + 1e-8)
                    area = self.connect_area[cnt_conn_index] / (self.scale ** 2)

                    if 0.95 >= regurgitation_scale >= 0.05 and area > minArea:
                        k = np.ones((3, 3))
                        draw_rect = cv2.dilate(check_regurgitation, k, iterations=5)
                        cnt_rect, _ = cv2.findContours(draw_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(draw_img, cnt_rect, -1, (0, 255, 0), 2)

                        # center, wh, theta = rect
                        # length, width = wh
                        points_y = sorted(points[:, 1])
                        depth = points_y[2] - points_y[0]
                        data_info.append((depth, area))

                        # 給 txt 資訊, 和 時間條
                        self.effect_area.append(round(area, 2))
                        self.que_frame.append(self.curr_frame_count)

                else:
                    reg_data_info = list()
                    # curr: 用最小擬合矩形框出
                    rect = cv2.minAreaRect(cnt_connect[cnt_conn_index])
                    points = cv2.boxPoints(rect).astype(np.int0)

                    # 判斷該區域是否有逆流
                    cv2.drawContours(check_regurgitation, [points], -1, (255, 255, 255), -1)
                    check_frame = frame.copy()
                    check_frame[check_regurgitation != 255] = [0, 0, 0]
                    check_hsv = cv2.cvtColor(check_frame, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(check_hsv)
                    h_min, h_max = self.HRange
                    h[h < h_min] = 0
                    h[h > h_max] = 0

                    uni_h, count = np.unique(h, return_counts=True)
                    uni_h, count = uni_h[1:], count[1:]  # 去除 0 的部分
                    blue_count = np.sum(count[uni_h > 77])
                    red_count = np.sum(count[uni_h < 35])

                    # 將比例設定在 0, 1 之間
                    regurgitation_scale = np.minimum(red_count, blue_count) / (np.maximum(red_count, blue_count) + 1e-8)

                    # curr: 有複數框時 找出最大面積供判斷 嚴重程度
                    reg_effect_area = list()

                    if 0.95 >= regurgitation_scale >= 0.05:
                        # 判斷圈出來的方框分別對應於未相連輪廓的位置(一樣以中心點計算)
                        M = cv2.moments(cnt_connect[cnt_conn_index])
                        conn_x = int(M["m10"] / M["m00"])
                        conn_y = int(M["m01"] / M["m00"])

                        curr_dis_info = list()
                        for pos_index in range(len(self.all_center)):
                            ori_x, ori_y = self.all_center[pos_index]
                            dis = np.sqrt((ori_x - conn_x) ** 2 + (ori_y - conn_y) ** 2)
                            curr_dis_info.append(dis)

                        min_index = curr_dis_info.index(min(curr_dis_info))

                        for element in self.connect_index:
                            if min_index in element:
                                curr_pos_index = self.connect_index.index(element)
                                area = self.connect_area[curr_pos_index] / (self.scale ** 2)

                                if area > minArea:
                                    # center, wh, theta = rect
                                    # length, width = wh

                                    k = np.ones((3, 3))
                                    draw_rect = cv2.dilate(check_regurgitation, k, iterations=5)
                                    cnt_rect, _ = cv2.findContours(draw_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(draw_img, cnt_rect, -1, (0, 255, 0), 2)

                                    points_y = sorted(points[:, 1])
                                    depth = points_y[2] - points_y[0]

                                    reg_data_info.append((depth, area))
                                    reg_effect_area.append(area)

                    if len(reg_effect_area) > 0:
                        max_area = max(reg_effect_area)
                        ori_index = reg_effect_area.index(max_area)
                        self.effect_area.append(round(max_area, 2))
                        data_info.append(reg_data_info[ori_index])
                        self.que_frame.append(self.curr_frame_count)

        # 顯示文字
        # 判斷分級程度
        pred_diag = self.degree()
        if len(self.que_degree) == 0:
            cv2.putText(draw_img, 'Level: Normal', (20, 525), font, 0.8, (255, 255, 255), 1)

        else:
            min_index = min(self.que_degree)
            level = self.degree_name[min_index]
            cv2.putText(draw_img, 'Level: %s' % level, (20, 525), font, 0.8, (255, 255, 255), 1)

        if len(data_info) == 0:
            cv2.putText(draw_img, 'Area:            , Depth:   ', (20, 555), font, 0.8, (255, 255, 255), 1)

        else:
            depth, area = data_info[0]
            depth = round(depth / self.scale, 2)
            area = round(area, 2)

            text_info = 'Area: %.2f sq.cm, Depth: %.2f cm' % (area, depth)
            cv2.putText(draw_img, text_info, (20, 555), font, 0.8, (255, 255, 255), 1)

            # text_l = 'width: %.2f cm' % length
            # text_w = 'deep: %.2f cm' % width
            # text_area = 'area: %.2f sq.cm' % area
            # cv2.putText(draw_img, text_l, (x_st, y_st + 3 * h * i), font, 0.8, (0, 255, 0), 1)
            # cv2.putText(draw_img, text_w, (x_st, y_st + h + 3 * h * i), font, 0.8, (0, 255, 0), 1)
            # cv2.putText(draw_img, text_area, (x_st, y_st + 2 * h + 3 * h * i), font, 0.8, (0, 255, 0), 1)

        if personalInfo:
            DCM = DicomData(dcm_path)
            text_name = 'Name: %s' % DCM.name
            text_birth = 'Birth: %s' % DCM.Birth
            text_age = 'Age: %d' % DCM.age
            text_sex = 'Sex: %s' % DCM.sex

            cv2.putText(draw_img, text_name, (90, 75), font, 0.7, (0, 255, 0), 1)
            cv2.putText(draw_img, text_birth, (90, 100), font, 0.7, (0, 255, 0), 1)
            cv2.putText(draw_img, text_age, (90, 125), font, 0.7, (0, 255, 0), 1)
            cv2.putText(draw_img, text_sex, (90, 150), font, 0.7, (0, 255, 0), 1)

        curr_res = cv2.addWeighted(frame, 1, draw_img, 1, 1)
        return curr_res

    def show_TimeBar(self, result_list):
        """
        parameter:
            已經處理過的 result

        return: list
        """
        curr_res = list()
        time_bar_img = np.zeros(result_list[0].shape, np.uint8)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # cv2.putText(time_bar_img, '1', (486, 145), font, 0.7, (255, 255, 255), 1)
        # cv2.putText(time_bar_img, str(self.total_frame_count), (680, 145), font, 0.7, (255, 255, 255), 1)
        interval = 200 / self.total_frame_count

        # 有問題區域標記
        time_bar_img[100:110, 490:700] = [255, 255, 255]
        for i in range(len(self.que_frame)):
            lx = np.floor(interval * self.que_frame[i] + 490).astype(np.int0)
            rx = np.floor(interval * self.que_frame[i] + 500).astype(np.int0)

            other = self.que_degree[i] * 50
            time_bar_img[100:110, lx:rx] = [other, other, 255]

        # 顯示 color bar
        cv2.putText(time_bar_img, 'Normal', (655, 195), font, 0.7, (255, 255, 255), 1)
        cv2.putText(time_bar_img, 'Severe', (660, 55), font, 0.7, (255, 255, 255), 1)
        time_bar_img[50:80, 720:740] = [0, 0, 255]
        time_bar_img[80:110, 720:740] = [50, 50, 255]
        time_bar_img[110:140, 720:740] = [100, 100, 255]
        time_bar_img[140:170, 720:740] = [150, 150, 255]
        time_bar_img[170:200, 720:740] = [255, 255, 255]

        time_bar_img[110:115, 490:700] = [255, 255, 0]

        # 時間條
        time_bar_img[115:130, 490:700] = [255, 255, 255]
        for i in range(1, self.total_frame_count+1):
            if i > 0:
                lx = int(interval * (i-1) + 490)
                rx = int(interval * (i-1) + 500)
                time_bar_img[115:130, lx:rx] = [255, 255, 255]

            lx = int(interval * i + 490)
            rx = int(interval * i + 500)
            time_bar_img[115:130, lx:rx] = [0, 255, 0]
            res = cv2.addWeighted(result_list[i-1], 1, time_bar_img, 1, 1)
            curr_res.append(res)

        return curr_res

    def writeVideo(self, frames, output_path, fps=30):
        """
        function:
            writeVideo(self, frames, output_path[, fps=30]): 將影片寫到指定路徑

        parameter:
            frames: 儲存所有 frame 的列表, list
            output_path: 影片輸出路徑(沒有階層資料夾), str
            fps: 每秒顯示幀數, int
        """
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (self.width, self.height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

    def gen_txt(self, txt_path):
        with open(txt_path, 'a+') as f:
            f.write(str(self.effect_area) + '\n')
        f.close()

    def degree(self):
        """
        wait
        img_path: image output path
        """
        interval, count = np.unique(self.effect_area, return_counts=True)
        degree_name = ['severe', 'moderate', 'mild', 'trivial']
        max_index = None

        if self.diag_pos == 'Aortic':
            # 顯示在影像上
            for i in range(len(self.effect_area)):
                if self.effect_area[i] >= self.aortic_Q75:
                    self.que_degree.append(0)

                elif self.aortic_Q75 > self.effect_area[i] >= self.aortic_Q50:
                    self.que_degree.append(1)

                elif self.aortic_Q50 > self.effect_area[i] >= self.aortic_Q25:
                    self.que_degree.append(2)

                else:
                    self.que_degree.append(3)

            # 預測診斷結果 (對應文字報告)
            trivial_count = np.sum(count[interval < self.aortic_Q25])
            mild_count = np.sum(count[interval < self.aortic_Q50]) - trivial_count
            moderate_count = np.sum(count[interval < self.aortic_Q75]) - mild_count - trivial_count
            severe_count = np.sum(count[interval >= self.aortic_Q75])

            count_list = [severe_count, moderate_count, mild_count, trivial_count]
            max_index = count_list.index(max(count_list))

        elif self.diag_pos == 'Mitral':
            # 顯示在影像上
            for i in range(len(self.effect_area)):
                if self.effect_area[i] >= self.mitral_Q75:
                    self.que_degree.append(0)

                elif self.mitral_Q75 > self.effect_area[i] >= self.mitral_Q50:
                    self.que_degree.append(1)

                elif self.mitral_Q50 > self.effect_area[i] >= self.mitral_Q25:
                    self.que_degree.append(2)

                else:
                    self.que_degree.append(3)

            # 預測診斷結果 (對應文字報告)
            trivial_count = np.sum(count[interval < self.mitral_Q25])
            mild_count = np.sum(count[interval < self.mitral_Q50]) - trivial_count
            moderate_count = np.sum(count[interval < self.mitral_Q75]) - mild_count - trivial_count
            severe_count = np.sum(count[interval >= self.mitral_Q75])

            count_list = [severe_count, moderate_count, mild_count, trivial_count]
            max_index = count_list.index(max(count_list))

        elif self.diag_pos == 'Pulmonary':
            # 顯示在影像上
            for i in range(len(self.effect_area)):
                if self.effect_area[i] >= self.pulmonary_Q75:
                    self.que_degree.append(0)

                elif self.pulmonary_Q75 > self.effect_area[i] >= self.pulmonary_Q50:
                    self.que_degree.append(1)

                elif self.pulmonary_Q50 > self.effect_area[i] >= self.pulmonary_Q25:
                    self.que_degree.append(2)

                else:
                    self.que_degree.append(3)

            # 預測診斷結果 (對應文字報告)
            trivial_count = np.sum(count[interval < self.pulmonary_Q25])
            mild_count = np.sum(count[interval < self.pulmonary_Q50]) - trivial_count
            moderate_count = np.sum(count[interval < self.pulmonary_Q75]) - mild_count - trivial_count
            severe_count = np.sum(count[interval >= self.pulmonary_Q75])

            count_list = [severe_count, moderate_count, mild_count, trivial_count]
            max_index = count_list.index(max(count_list))

        elif self.diag_pos == 'Tricuspid':
            # 顯示在影像上
            for i in range(len(self.effect_area)):
                if self.effect_area[i] >= self.tricuspid_Q75:
                    self.que_degree.append(0)

                elif self.tricuspid_Q75 > self.effect_area[i] >= self.tricuspid_Q50:
                    self.que_degree.append(1)

                elif self.tricuspid_Q50 > self.effect_area[i] >= self.tricuspid_Q25:
                    self.que_degree.append(2)

                else:
                    self.que_degree.append(3)

            # 預測診斷結果 (對應文字報告)
            trivial_count = np.sum(count[interval < self.tricuspid_Q25])
            mild_count = np.sum(count[interval < self.tricuspid_Q50]) - trivial_count
            moderate_count = np.sum(count[interval < self.tricuspid_Q75]) - mild_count - trivial_count
            severe_count = np.sum(count[interval >= self.tricuspid_Q75])

            count_list = [severe_count, moderate_count, mild_count, trivial_count]
            max_index = count_list.index(max(count_list))

        self.level = degree_name[max_index]
        return degree_name[max_index]


class DicomData(object):
    def __init__(self, dcm_path):
        self.path = dcm_path
        self.dcm = pydicom.dcmread(dcm_path)

        # dcm Data 屬性
        self.id = self.dcm.PatientID
        self.name = self.dcm.PatientName
        self.Birth = self.dcm.PatientBirthDate
        self.sex = self.dcm.PatientSex
        self.study_date = self.dcm.StudyDate
        self.study_time = self.dcm.StudyTime

        # 計算 age 用
        self.local_time = time.localtime()
        self.str_localtime = time.strftime('%Y %m %d %H %M %S %p', self.local_time)
        self.local_year = int(self.str_localtime.split(' ')[0])
        self.local_month = int(self.str_localtime.split(' ')[1])
        self.local_day = int(self.str_localtime.split(' ')[2])

        self.year, self.month, self.day = int(self.Birth[:4]), int(self.Birth[4:6]), int(self.Birth[6:])
        self.__m = self.month > self.local_month
        self.__d = self.month == self.local_month and self.day > self.local_day

        self.age = self.local_year - self.year - 1 if self.__m or self.__d else self.local_year - self.year

    def conv_avi(self):
        pass
