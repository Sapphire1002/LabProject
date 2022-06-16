from pandas import DataFrame
import numpy as np
import os
import cv2
# import time


def read_file(file_path):
    """
    function:
        read_file: 讀取 avi 檔案

    parameter:
        file_path: 放資料夾路徑

    return:
        case_avi_dict: 傳回 所有病例的 avi 檔案 dict
    """
    all_case_dir = os.listdir(file_path)
    all_case_dir_path = list()
    case_avi_dict = dict()

    for case_dir in all_case_dir:
        case_dir_path = os.path.join(path, case_dir) + '/'
        all_case_dir_path.append(case_dir_path)

    for case_path in all_case_dir_path:
        curr_case = case_path.split('/')[-2]
        case_avi_dict[curr_case] = list()

        dir_case = os.listdir(case_path)

        # 過濾空資料夾
        if len(dir_case) > 0:
            for case_category in dir_case:
                case_category_path = case_path + case_category

                # 讀取 Mitral, tricuspid, Aortic, pulmonary 資料夾, 並且跳過 txt 檔
                if os.path.isdir(case_category_path):
                    case_category_path = case_category_path + '/'
                    case_category_files = os.listdir(case_category_path)

                    # 四種類別底下有檔案的情況
                    if len(case_category_files) > 0:
                        for case_avi in case_category_files:
                            case_avi_path = case_category_path + case_avi
                            case_avi_dict[curr_case].append(case_avi_path)
    return case_avi_dict


def write_video(frames_list, output_path):
    y, x, _ = frames_list[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))
    for frames in frames_list:
        video_writer.write(frames)
    video_writer.release()


def handle_doppler(case_file_path):
    """
    function:
        handle_doppler: 針對 Doppler 區域抓取有問題的區域

    parameter:
        case_file_path: 病例檔案路徑

    return:
        bool_value: 用來讓程式提早結束
        excel_area_data: 寫入 excel 需要的資料
        result: 每一幀抓取有問題區域後的結果
    """
    # 類別
    # category = ['Aortic', 'Mitral', 'Pulmonary', 'Tricuspid']  # 主動脈, 二尖瓣, 肺動脈, 三尖瓣

    # 儲存結果的 frames 列表
    result = list()

    # 儲存寫入 excel 的資料
    excel_area_data = list()

    # 讀取該病例的 avi 檔案
    video = cv2.VideoCapture(case_file_path)

    # 避免都卜勒扇形區域破損造成影像缺失
    _, frame = video.read()
    old_img_contours = np.zeros(frame.shape[:2], np.uint8)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # 最原始的 frame
        original_frame = frame.copy()

        # 1. 找出扇形 Doppler 區域 (目前沒有問題)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 211]), np.array([180, 30, 255]))

        kernel = np.ones((3, 3), np.uint8)
        mask_white_dilate = cv2.dilate(mask_white, kernel=kernel, iterations=1)

        contours, hier = cv2.findContours(mask_white_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(frame.shape[:2], np.uint8)

        for c in contours:
            area = cv2.contourArea(c)
            cnt_len = cv2.arcLength(c, closed=False)

            if area > 40000 or cnt_len > 500:
                cv2.drawContours(img_contours, [c], -1, (255, 255, 255), -1)

                # 若 img_contour 的面積低於 30000, 代表扇形以內的區塊有極大機率沒有被填滿白色實心
                if np.unique(img_contours, return_counts=True)[1][1] < 30000:
                    img_contours = old_img_contours

                else:
                    old_img_contours = img_contours

        erode = cv2.erode(img_contours, kernel, iterations=2)
        frame[erode != 255] = [0, 0, 0]

        # 2. 找出有問題的區域並圈出來後, 計算面積及長寬(目前 OK)
        # 最後結果的矩形框畫在 draw_img 上面
        draw_img = np.zeros(frame.shape, np.uint8)

        # 找出有問題區域的顏色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_others = cv2.inRange(hsv, np.array([20, 43, 46]), np.array([99, 255, 255]))

        # 濾掉雜點(開運算)
        mask_others = cv2.morphologyEx(mask_others, cv2.MORPH_OPEN, kernel=kernel)
        contour_others, _ = cv2.findContours(mask_others, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 計算每個輪廓的區域後, 過濾比平均值還小的面積
        all_area = list()

        # 計算每個輪廓的面積
        for cnt in contour_others:
            area = cv2.contourArea(cnt)
            all_area.append(area)

        # 1e-8 避免被 0 除的問題, 接著將低於平均值的區域在 mask_others 塗黑後, 再找出 contour
        area_avg = sum(all_area) / (len(all_area) + 1e-8)
        for filter_index in range(len(all_area)):
            # 使用平均值來過濾掉過小的區塊
            if all_area[filter_index] <= area_avg:
                cv2.drawContours(mask_others, [contour_others[filter_index]], -1, (0, 0, 0), -1)

        mask_last = mask_others.copy()  # 計算點距離時會用到

        # 註: 這裡的 contour_others_filter 是尚未將輪廓全部畫起來的(所以面積要在這邊計算)
        contour_filter, _ = cv2.findContours(mask_others, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 過濾小區塊後, 計算有問題區域的面積
        filter_area = np.zeros(len(contour_filter))
        for cnt_filter_index in range(len(contour_filter)):
            area = cv2.contourArea(contour_filter[cnt_filter_index])
            filter_area[cnt_filter_index] = area

        # 此時所有面積的總和是 excel 要呈現的資料
        excel_area_data.append(np.sum(filter_area))

        # ----- 距離 & 面積計算 -----
        # 儲存所有輪廓的近似中心點
        all_center = list()

        # 利用 minAreaRect (), 找出近似輪廓的中心點
        for cnt_center in contour_filter:
            rect = cv2.minAreaRect(cnt_center)
            center, wh, theta = rect
            all_center.append(center)

        # 計算每個中心點的距離後, 判斷是否連接
        connect_index = list()  # 儲存連接的輪廓索引

        for cnt_init_index in range(len(all_center)):
            x_init, y_init = all_center[cnt_init_index]

            # print('cnt_init_index:', cnt_init_index)
            curr_index = list()
            curr_index.append(cnt_init_index)

            # 判斷 cnt_init_index 的值是否已被其他點連接, 若已被連接則由已連接的區域繼續往下延伸
            if len(connect_index) > 0:
                for is_connect_index in range(len(connect_index)):
                    if cnt_init_index in connect_index[is_connect_index]:
                        curr_index = connect_index[is_connect_index]

            # print('1st loop curr_index:', curr_index)

            for cnt_next_index in range(cnt_init_index + 1, len(all_center)):
                x_end, y_end = all_center[cnt_next_index]
                dis = np.sqrt((x_end - x_init) ** 2 + (y_end - y_init) ** 2)

                # 避免中心點是黑色區域(暫時用這方法處理)
                # 若中心點是黑色區域, 則找該輪廓的第一個點位置做連接
                if mask_others[int(y_end), int(x_end)] != 255:
                    x_end, y_end = contour_filter[cnt_next_index][0][0]

                # 判斷兩點距離(暫時設定 60)
                if dis <= 60:
                    # 若下個節點, 沒有被相連過, 就加到 curr_index 裡面
                    if cnt_next_index not in curr_index:
                        curr_index.append(cnt_next_index)

                    x_init, y_init, x_end, y_end = int(x_init), int(y_init), int(x_end), int(y_end)
                    cv2.line(mask_last, (x_init, y_init), (x_end, y_end), (255, 255, 255), 1)

            # print('2nd loop curr_index:', curr_index)
            sort_index = sorted(curr_index)
            # print('sort_index:', sort_index)

            # 避免加入重複的元素
            if sort_index not in connect_index:
                connect_index.append(sort_index)
                # print('connect_index:', connect_index)

        # 計算連接輪廓後的面積
        connect_area = list()  # 儲存連接的面積
        for count in connect_index:
            area = 0
            for index in count:
                area += filter_area[index]
            connect_area.append(area)

        # print('filter area:', filter_area)
        # print('after handle connect index:', connect_index)
        # print('connect area:', connect_area)

        # cv2.imshow('mask_others', mask_others)
        # cv2.imshow('mask_last', mask_last)
        # ----- End -----

        # 3. 將輪廓的區域圈出來並顯示相對應的面積和長寬
        data_info = list()  # 儲放當前 frame, 的資訊(長, 寬, 面積)
        color_info = list()  # 輪廓不只一個時需要用不同顏色區分

        contour_connect, _ = cv2.findContours(mask_last, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print('contour_connect length:', len(contour_connect))

        # 找出相對應的面積和長寬
        for cnt_connect_index in range(len(contour_connect)):
            if len(contour_connect) == 1:
                # 把輪廓區域用最小擬合矩形框框出來
                rect = cv2.minAreaRect(contour_connect[cnt_connect_index])
                points = cv2.boxPoints(rect).astype(np.int0)
                cv2.drawContours(draw_img, [points], -1, (0, 255, 0), 2)

                # 顯示文字的資訊
                center, wh, theta = rect
                length, width = wh
                color_info.append((0, 255, 0))
                data_info.append((length, width, connect_area[cnt_connect_index]))

            # 代表輪廓任一處有相連的部分
            else:
                rect = cv2.minAreaRect(contour_connect[cnt_connect_index])
                points = cv2.boxPoints(rect).astype(np.int0)

                b = np.random.randint(50, 255)
                g = np.random.randint(50, 220)
                r = np.random.randint(50, 255)

                cv2.drawContours(draw_img, [points], -1, (b, g, r), 2)

                # 要判斷圈出來的方框分別對應於未相連前輪廓的哪個位置
                center, wh, theta = rect
                length, width = wh

                # 判斷點是否在框的內部
                curr_dis_info = list()

                for cnt_pos_index in range(len(all_center)):
                    center_x, center_y = center
                    pos_x, pos_y = all_center[cnt_pos_index]
                    dis = np.sqrt((center_x - pos_x) ** 2 + (center_y - pos_y) ** 2)
                    curr_dis_info.append(dis)

                min_dis_index = curr_dis_info.index(min(curr_dis_info))
                curr_pos_index = None

                # 判斷最小的距離索引值, 位於 connect_index 的哪個位置
                for element in connect_index:
                    if min_dis_index in element:
                        curr_pos_index = element
                try:
                    index_pos = connect_index.index(curr_pos_index)
                    color_info.append((b, g, r))
                    data_info.append((length, width, connect_area[index_pos]))

                except ValueError:
                    print('Current Frame Error.')
                    # print('connect_index:', connect_index)
                    # print('curr_pos_index:', curr_pos_index)
                    cv2.imshow('original_frame', original_frame)
                    cv2.imshow('draw_img', draw_img)
                    cv2.imshow('mask_others', mask_others)
                    cv2.imshow('mask_last', mask_last)
                    cv2.waitKey(0)

        # 顯示文字部分
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        x_st, y_st = (90, 100)
        i, h = 0, 25

        for length, width, area in data_info:
            text_lw = 'length: %d, width: %d' % (length, width)
            text_area = 'area: %.1f' % area

            cv2.putText(original_frame, text_lw, (x_st, y_st + 2 * h * i), font, 0.6, color_info[i], 1)
            cv2.putText(original_frame, text_area, (x_st, y_st + h + 2 * h * i), font, 0.6, color_info[i], 1)
            i += 1

        original_frame = cv2.addWeighted(original_frame, 1, draw_img, 0.6, 0)
        result.append(original_frame)

        # cv2.imshow('original_frame', original_frame)
        # cv2.imshow('draw_img', draw_img)

        key = cv2.waitKey(10)
        if key == ord('q'):
            return 0, excel_area_data, result

        elif key == ord('p'):
            while cv2.waitKey(1) != ord(' '):
                pass
    return 1, excel_area_data, result


def write_excel(data, case_num, output_excel_path):
    """
    function:
        write_excel: 將資料寫入 excel

    parameter:
        data: 要寫入 excel 的資料
        case_num: 病例名稱
        output_excel_path: 輸出的路徑

    return:
        None
    """

    df = DataFrame(data.values(), index=data.keys()).T
    df.to_excel(output_excel_path, sheet_name=case_num)


if __name__ == '__main__':
    path = './video/1st data/'
    case_avi_files = read_file(path)

    # case_name 為 病歷號碼
    for case_name in case_avi_files.keys():
        excel_data = dict()
        # excel_path = './video/1st data/' + case_name + '/' + case_name + '.xlsx'

        for avi_path in case_avi_files[case_name]:
            output_dir = './video/1st data result/' + avi_path[len(path):]
            print(avi_path)
            stop, area_data, res = handle_doppler(avi_path)
            # if stop == 0:
            #     break
            #
            # else:
            #     avi_file_name = avi_path.split('/')[-1]
            #     if avi_file_name not in excel_data.keys():
            #         excel_data[avi_file_name] = area_data
            write_video(res, output_dir)
        # write_excel(excel_data, case_name, excel_path)
