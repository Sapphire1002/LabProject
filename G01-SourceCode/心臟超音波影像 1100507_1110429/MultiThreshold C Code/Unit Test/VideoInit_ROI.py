import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def AllFiles(DirPath, extension_name='avi'):
    result = list()
    for root, dirs, files in os.walk(DirPath):
        for f in files:
            if f[-len(extension_name):] == extension_name:
                result.append(os.path.join(root, f))
    return result


class VideoInit(object):
    """
    Class name:
        VideoInit(object):
        處理影片的基本資訊
        目前: ROI、心臟範圍、標準單位及BPM(測試階段)

    Initialization parameters:
        VideoPath: 輸入影片路徑, str
    """
    def __init__(self, VideoPath):
        # 使用 cv2 讀取影片
        self.video = cv2.VideoCapture(VideoPath)

        if not self.video:
            raise FileNotFoundError('檔案路徑不存在或影片無法讀取')

        else:
            # _ROI 屬性
            self.roi = self._ROI(VideoPath)

            # _Unit 屬性(Test, 尚未完成)
            self._unit = None
            self._bpm = None

            # HeartBound 屬性
            self.dtC, self.dtM = None, None
            self.maskC, self.maskM = None, None
            self.contourC, self.contourM = None, None

            # hist 屬性
            self.grayScale = None
            self.count = None

    def _ROI(self, Path):
        """
        method name:
            _ROI(Path):
            找到超音波影像的有效區域

        parameters:
            Path: 影片檔案路徑, str

        return:
            roi: 超音波有效區域二值化圖片, numpy.ndarray, 大小為 (height, width)

        attributes:
            self.ox, self.oy: ROI 扇形區域的中心點 x, y 軸座標, int
            self.radius: ROI 扇形的半徑, int
        """
        # ----- 1. 找出超音波影像有效區域
        target = cv2.VideoCapture(Path)
        _, first = target.read()

        # self._Unit(first)  # 處理標準單位 & BPM (測試階段)

        gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

        # 疊加所有差幀結果
        mask_diff_all = np.zeros(gray_first.shape, np.uint8)

        while True:
            _ret, f = target.read()

            if not _ret:
                break

            gray_f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_first, gray_f)

            diff[diff > 10] = 255
            diff[diff <= 10] = 0
            mask_diff_all += diff
            np.clip(mask_diff_all, 0, 255, out=mask_diff_all)

        mask_last = np.zeros(gray_first.shape, np.uint8)
        cnt, _ = cv2.findContours(mask_diff_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_last, cnt, -1, (255, 255, 255), -1)

        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(mask_last, kernel, iterations=3)
        dilate = cv2.dilate(erode, kernel, iterations=2)

        mask_last_bound = np.zeros(gray_first.shape, np.uint8)
        cnt, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt:
            if cv2.contourArea(c) >= 300:
                cv2.drawContours(mask_last_bound, [c], -1, (255, 255, 255), 2)
        cnt_last, _ = cv2.findContours(mask_last_bound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # ----- mask ROI End.

        # ----- 2. 找出有效區域的圓心座標和半徑 & 繪製扇形 mask(避免有鋸齒或影像不完整導致 ROI 效果不好)
        roi = np.zeros(first.shape[:2], np.uint8)

        # 霍夫轉換找直線求兩線交點
        lines = cv2.HoughLinesP(
            mask_last_bound, 1, np.pi / 180,
            threshold=200,
            minLineLength=60,
            maxLineGap=130
        )

        x1_y1, x2_y2 = list(), list()
        lm_error, rm_error = 1, 1
        l_index, r_index = None, None  # 儲存兩條線最接近斜率為 -1, 1 的索引值

        try:
            for line_index in range(len(lines)):
                x1, y1, x2, y2 = lines[line_index][0]
                m = (y2 - y1) / ((x2 - x1) + 1e-08)  # 1e-08 避免分母為 0

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

            # 利用方程式求出圓心座標
            a1, b1 = x1_y1[l_index]
            a2, b2 = x2_y2[l_index]
            m1 = (b2 - b1) / (a2 - a1)

            A1, B1 = x1_y1[r_index]
            A2, B2 = x2_y2[r_index]
            m2 = (B2 - B1) / (A2 - A1)

            c0, c1 = m1 * a1 - b1, m2 * A1 - B1

            ox = np.round((c0 - c1) / (m1 - m2)).astype(np.int)
            oy = np.round(((m1 + m2) * ox - c0 - c1) / 2).astype(np.int)

            # 找半徑
            radius = 0
            for i in range(len(cnt_last)):
                for j in range(len(cnt_last[i])):
                    if radius < cnt_last[i][j][0][1]:
                        radius = cnt_last[i][j][0][1]
            radius = radius - oy
            cv2.ellipse(roi, (ox, oy), (radius, radius), 90, -45, 45, (255, 255, 255), -1)

        except TypeError:
            # 霍夫轉換找不到直線時, 拿 y 軸的最小值當圓心
            radius = 0
            ox, oy = 0, 600

            for i in range(len(cnt_last)):
                for j in range(len(cnt_last[i])):
                    if radius < cnt_last[i][j][0][1]:
                        radius = cnt_last[i][j][0][1]

                    if oy > cnt_last[i][j][0][1]:
                        ox, oy = cnt_last[i][j][0]
            radius = radius - oy
            cv2.ellipse(roi, (ox, oy), (radius, radius), 90, -45, 45, (255, 255, 255), -1)

        self.ox, self.oy, self.radius = ox, oy, radius
        return roi

    def _Unit(self, img):
        """
        (Test... 尚未完成)
        method name:
            _Unit(img):
            找出標準單位和 bpm 數值

        parameter:
            img: 輸入第一幀影像
        """
        # ----- Pre. 每個數字對應的 KeyValue
        # {(min_index, max_index): math}
        math_key = {
            (2, 4): 2,
            (2, 3): 3,
            (2, 5): 3,
            (0, 5): 4,
            (1, 0): 5,
            (1, 2): 6,
            (5, 1): 7,
            (5, 2): 8,
        }

        # ----- 1. 找標準單位
        unit_pos = img[75:93, 19:29]
        unit_pos = cv2.cvtColor(unit_pos, cv2.COLOR_BGR2GRAY)
        unit_pos = cv2.resize(unit_pos, (40, 72), cv2.INTER_CUBIC)
        unit_pos[unit_pos > 128] = 255
        unit_pos[unit_pos <= 128] = 0

        # 找出最小矩形框切成 6 等分 計算白色區域面積
        cnt_unit, _ = cv2.findContours(unit_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        unit_rect = cv2.boundingRect(cnt_unit[0])
        x1, y1, width, height = unit_rect

        # 分成六等分
        h = height // 3
        w = width // 2
        cx, ucy, dcy = x1 + w, y1 + h, y1 + height - h

        # 計算白色區域面積
        ul_score = np.unique(unit_pos[y1:ucy, x1:cx], return_counts=True)[1][1]
        cl_score = np.unique(unit_pos[ucy:dcy, x1:cx], return_counts=True)[1][1]
        dl_score = np.unique(unit_pos[dcy:y1 + height, x1:cx], return_counts=True)[1][1]

        ur_score = np.unique(unit_pos[y1:ucy, cx:x1 + width], return_counts=True)[1][1]
        cr_score = np.unique(unit_pos[ucy:dcy, cx:x1 + width], return_counts=True)[1][1]
        dr_score = np.unique(unit_pos[dcy:y1 + height, cx:x1 + width], return_counts=True)[1][1]

        score_list = [ul_score, ur_score, cl_score, cr_score, dl_score, dr_score]

        # Unit 的數值
        min_index = score_list.index(min(score_list))
        max_index = score_list.index(max(score_list))
        key_unit = (min_index, max_index)
        self.unit = math_key[key_unit] + 10
        # ----- 1. 找標準單位 End.

        # ----- 2. 找 bpm

        pass

    def HeartBound(self, img, imgSize=None, isBlur=False, KBlur=9, maxVal=255, AlgThres=cv2.ADAPTIVE_THRESH_MEAN_C,
                   BlockSize=131, C=9, AlgDistance=cv2.DIST_L1, DistanceMaskSize=5):
        """
        method name:
            HeartBound(img, imgSize=None, isBlur=False, KBlur=9, maxVal=255, AlgThres=cv2.ADAPTIVE_THRESH_MEAN_C,
                       BlockSize=131, C=9, AlgDistance=cv2.DIST_L1, DistanceMaskSize=5):
            將圖像做自適應二值化 (adaptive threshold) 和距離變換 (distance transform) 後找出心臟範圍

        parameters:
            img: 原始影像或灰階影像, numpy.ndarray
            imgSize: 調整圖像大小, tuple, (height, width), 默認 None 代表原始圖像的大小
            isBlur: 輸入圖像是否濾波, bool, 默認 False 代表尚未濾波
            KBlur: medianBlur 的 Kernel Size, int, 默認 9
            maxVal: adaptiveThreshold 參數, 二值化最大值, int, 默認 255
            AlgThres: adaptiveThreshold 參數, 演算法類型, 默認平均值 (cv2.ADAPTIVE_THRESH_MEAN_C)
            BlockSize: adaptiveThreshold 參數, 參考的局部大小, int, 默認 131
            C: adaptiveThreshold 參數, 偏移量, int, 默認 9
            AlgDistance: distanceTransform 參數, 計算距離的方式, 默認曼哈頓距離 (cv2.DIST_L1)
            DistanceMaskSize: distanceTransform 參數, 距離計算的範圍大小, int, 默認 5

        attributes:
            self.dtC: 腔室距離變換的圖像, 資料類型 numpy.uint8
            self.dtM: 肌肉距離變換的圖像, 資料類型 numpy.uint8
            self.maskC: 腔室區域的二值化 mask
            self.maskM: 肌肉區域的二值化 mask
            self.contourC: 腔室區域的輪廓
            self.contourM: 肌肉區域的輪廓
        """

        # ----- Pre. 圖像濾波
        # 判斷 img 是否已經濾波或轉灰階
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not isBlur:
            img = cv2.medianBlur(img, KBlur)

        # ----- 1. 取出腔室和肌肉的自適應二值化 & 距離變換
        kernel = np.ones((5, 5), np.uint8)

        # 腔室部分
        ChamberThres = cv2.adaptiveThreshold(img, maxVal, AlgThres, cv2.THRESH_BINARY_INV, BlockSize, C)
        OpeningC = cv2.morphologyEx(ChamberThres, cv2.MORPH_OPEN, kernel)

        DT_C = cv2.distanceTransform(OpeningC, AlgDistance, DistanceMaskSize).astype(np.float32)
        scale_C = 255 / np.max(DT_C)
        DT_C *= scale_C
        DT_C = cv2.convertScaleAbs(DT_C)
        DT_C[self.roi != 255] = 0

        # 肌肉部分
        MuscleThres = cv2.adaptiveThreshold(img, maxVal, AlgThres, cv2.THRESH_BINARY, BlockSize, C)
        OpeningM = cv2.morphologyEx(MuscleThres, cv2.MORPH_OPEN, kernel)

        DT_M = cv2.distanceTransform(OpeningM, AlgDistance, DistanceMaskSize).astype(np.float32)
        DT_M = cv2.convertScaleAbs(DT_M)
        DT_M[self.roi != 255] = 0
        scale_M = 255 / np.max(DT_M)
        DT_M = (DT_M * scale_M).astype(np.float32)
        DT_M = cv2.convertScaleAbs(DT_M)

        # 儲存原始大小的 Distance Transform 圖片
        self.dtC, self.dtM = DT_C, DT_M

        # ----- 1. 取出腔室和肌肉的自適應二值化 & 距離變換 End.

        # ----- 2. 找出心臟範圍
        # 是否要調整影像大小
        height, width = img.shape[:2] if imgSize is None else imgSize

        # 腔室部分(先做一次腔室區域當成 mask)
        ReDT_C = cv2.resize(DT_C, (width, height), cv2.INTER_CUBIC)
        mask_chamber = np.zeros((height, width), np.uint8)

        ptsC = list()
        xaxis_index = np.argmax(ReDT_C, axis=1)
        yaxis_index = np.argmax(ReDT_C, axis=0)

        for i, j in enumerate(yaxis_index):
            if i and j:
                ptsC.append([i, j])

        for i, j in enumerate(xaxis_index):
            if i and j:
                ptsC.append([j, i])

        ptsC = np.asarray(ptsC)
        ptsCHull = cv2.convexHull(ptsC)
        cv2.drawContours(mask_chamber, [ptsCHull], 0, (255, 255, 255), -1)
        cnt_C, _ = cv2.findContours(mask_chamber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 肌肉部分
        ReDT_M = cv2.resize(DT_M, (width, height), cv2.INTER_CUBIC)
        mask_muscle = np.zeros((height, width), np.uint8)

        ptsM = list()
        xaxis_index = np.argmax(ReDT_M, axis=1)
        yaxis_index = np.argmax(ReDT_M, axis=0)

        for i, j in enumerate(yaxis_index):
            if i and j:
                is_region = cv2.pointPolygonTest(cnt_C[0], (i, j), False)

                if is_region != -1:
                    ptsM.append([i, j])

        for i, j in enumerate(xaxis_index):
            if i and j:
                is_region = cv2.pointPolygonTest(cnt_C[0], (j, i), False)

                if is_region != -1:
                    ptsM.append([j, i])

        ptsM = np.asarray(ptsM)
        ptsMHull = cv2.convexHull(ptsM)
        cv2.drawContours(mask_muscle, [ptsMHull], 0, (255, 255, 255), -1)
        cnt_M, _ = cv2.findContours(mask_muscle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 若圖像有被調整過, 要還原成原本大小
        if imgSize is not None:
            ori_height, ori_width = img.shape[:2]
            mask_chamber = cv2.resize(mask_chamber, (ori_width, ori_height), cv2.INTER_LINEAR)
            mask_muscle = cv2.resize(mask_muscle, (ori_width, ori_height), cv2.INTER_LINEAR)

            cnt_C, _ = cv2.findContours(mask_chamber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt_M, _ = cv2.findContours(mask_muscle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.maskC, self.maskM = mask_chamber, mask_muscle
        self.contourC, self.contourM = cnt_C, cnt_M
        # ----- 2. 找出心臟範圍 End.

    def hist(self, img, Display=False):
        """
        method name:
            hist(img, Display=False):
            計算灰階直方圖, 以及繪製直方圖

        parameters:
            img: 輸出原始影像或灰階影像, numpy.ndarray
            Display: 是否顯示柱狀圖, bool, 默認 False 不顯示

        attributes:
            self.grayScale: 灰階值的分布(0 ~ 255), numpy.ndarray, int, 大小為 (256, 1)
            self.count: 每個灰階值的數量, numpy.ndarray, int, 大小為 (256, 1)
        """

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img[self.maskM != 255] = 0
        ZeroArea = np.unique(self.maskM, return_counts=True)[1][0]

        value, count = np.unique(img, return_counts=True)
        count[0] -= ZeroArea

        self.grayScale = value
        self.count = count

        if Display:
            plt.title('Curr frame Histogram')
            plt.xlabel('Gray Scale')
            plt.ylabel('Count')
            plt.bar(self.grayScale, self.count)
            plt.show()
