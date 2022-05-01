import cv2
import numpy as np


def adjust_scale(model_all, model_A, model_B, model_C, radius):
    """
    function:
        adjust_scale(model_all, model_A, model_B, model_C, radius):
            調整標準模型比例

    parameter:
        model_all: 完整的標準模型
        model_A: 模型的 A 區段
        model_B: 模型的 B 區段
        model_C: 模型的 C 區段
        radius: 骨架圖的有效半徑(skeletonize_bound 回傳值)

    method:
        1. 找出完整標準模型的最小擬合橢圓
        2. 找出橢圓的圓心和半徑後, 裁減圖像
        3. 計算和骨架圖的半徑比例後, 進行縮放
        4. resize A, B, C 三個區段的模型

    return:
        output_model_A: resize 後的 model_A
        output_model_B: resize 後的 model_B
        output_model_C: resize 後的 model_C

    """
    model_all_gray = cv2.cvtColor(model_all, cv2.COLOR_BGR2GRAY)
    _, model_all_thres = cv2.threshold(model_all_gray, 180, 255, cv2.THRESH_BINARY)
    cnt_model, _ = cv2.findContours(model_all_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    (x, y), radius_model = cv2.minEnclosingCircle(cnt_model[0])
    x, y, radius_model = int(x), int(y), int(radius_model)

    left, right = x - radius_model, x + radius_model
    top, bottom = y - radius_model, y + radius_model

    model_all = model_all[top:bottom, left:right]
    model_A = model_A[top:bottom, left:right]
    model_B = model_B[top:bottom, left:right]
    model_C = model_C[top:bottom, left:right]

    scale = radius / radius_model
    ori_height, ori_width = model_all.shape[:2]
    width, height = int(ori_width * scale), int(ori_height * scale)

    output_model_A = cv2.resize(model_A, (width, height))
    output_model_B = cv2.resize(model_B, (width, height))
    output_model_C = cv2.resize(model_C, (width, height))

    return output_model_A, output_model_B, output_model_C
