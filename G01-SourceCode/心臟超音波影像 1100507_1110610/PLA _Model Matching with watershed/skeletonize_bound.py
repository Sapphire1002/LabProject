import cv2


def skeleton_bound(file_name, roi):
    """
    function:
        skeleton_bound(file_name, roi):
            調整 roi 的邊界大小

    parameter:
        file_name: 骨架圖的檔案路徑, str
        roi: find_roi 的傳回值

    method:
        1. 將 ROI 區域以外的等於黑色
        2. 找出輪廓的邊界, 分別計算寬高的長度
        3. 用圓的方式來找出有效的邊界區域
        (半徑計算方式: int(min(rad_x, rad_y) + abs(rad_x - rad_y) // 2))

    return:
        top, bottom, left, right: 上下左右四個點的座標, int
        radius: 半徑大小(調整模型比例用), int
    """
    skeletonize_file = cv2.imread(file_name)
    skeletonize_file[roi != 255] = [0, 0, 0]
    gray_skeletonize = cv2.cvtColor(skeletonize_file, cv2.COLOR_BGR2GRAY)

    x_bound, y_bound = list(), list()
    cnt_skeleton, _ = cv2.findContours(gray_skeletonize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(cnt_skeleton)):
        for j in range(len(cnt_skeleton[i])):
            x_bound.append(cnt_skeleton[i][j][0][0])
            y_bound.append(cnt_skeleton[i][j][0][1])

    try:
        x = int((max(x_bound) + min(x_bound)) / 2)
        y = int((max(y_bound) + min(y_bound)) / 2)
        rad_x = (max(x_bound) - min(x_bound)) / 2
        rad_y = (max(y_bound) - min(y_bound)) / 2

        radius = int(min(rad_x, rad_y) + abs(rad_x - rad_y) // 2)
        left, right = x - radius, x + radius
        top, bottom = y - radius, y + radius
        return [top, bottom, left, right, radius]

    except ValueError:
        print('%s 骨架圖片可能是全黑的' % file_name)
        return None
