import cv2
import numpy as np

# 目前的做法 A 區段要放全域, 否則 matching 不到會有 bug(這部分可以考慮寫在一起)
best_theta_A = 0
best_horizontal_A, best_vertical_A = 0, 0


def match_muscleA(frame, A, v_range=range(15, 80, 5), h_range=range(-30, 30, 6), t_range=range(-10, 20, 5)):
    max_fitting_area_A = 0
    global best_theta_A
    global best_horizontal_A, best_vertical_A

    model_best_A = None
    best_fitting_A = np.zeros(A.shape, np.uint8)

    y, x, _ = frame.shape
    for vertical in v_range:
        for horizontal in h_range:
            trans_mat = np.array([[1, 0, horizontal], [0, 1, vertical]], np.float32)
            affine_A = cv2.warpAffine(A, trans_mat, (x, y))

            for theta in t_range:
                rotate_mat = cv2.getRotationMatrix2D((x / 2, y / 2), theta, 1)
                rotate_A = cv2.warpAffine(affine_A, rotate_mat, (x, y))

                fitting_A = cv2.bitwise_and(rotate_A, frame)
                fitting_area = np.sum(fitting_A)

                if fitting_area > max_fitting_area_A:
                    max_fitting_area_A = fitting_area
                    best_theta_A = theta
                    best_vertical_A = vertical
                    best_horizontal_A = horizontal

                    model_best_A = rotate_A
                    best_fitting_A = cv2.bitwise_and(rotate_A, frame)

    return model_best_A, best_fitting_A


def match_muscleB(frame, B, seg_B, v_range=range(-10, 10, 5), h_range=range(-50, 10, 5), t_range=range(-10, 10, 2)):
    max_fitting_area_B = 0
    model_best_B = None
    best_fitting_B = None

    y, x, _ = frame.shape
    for vertical in v_range:
        for horizontal in h_range:
            trans_mat = np.array([[1, 0, horizontal], [0, 1, vertical]], np.float32)
            affine_B = cv2.warpAffine(B, trans_mat, (x, y))

            for theta in t_range:
                rotate_mat = cv2.getRotationMatrix2D((x / 2, y / 2), theta, 1)
                rotate_B = cv2.warpAffine(affine_B, rotate_mat, (x, y))

                fitting_B = cv2.bitwise_and(rotate_B, seg_B)
                fitting_area = np.sum(fitting_B)

                if fitting_area > max_fitting_area_B:
                    max_fitting_area_B = fitting_area
                    model_best_B = rotate_B
                    best_fitting_B = cv2.bitwise_and(rotate_B, frame)

    # 避免該片段模型無法擬合
    if model_best_B is None:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        trans_mat = np.array([[1, 0, best_horizontal_A], [0, 1, best_vertical_A]], np.float32)
        affine_B = cv2.warpAffine(B, trans_mat, (x, y))

        rotate_mat = cv2.getRotationMatrix2D((x/2, y/2), best_theta_A, 1)
        model_best_B = cv2.warpAffine(affine_B, rotate_mat, (x, y))
        best_fitting_B = model_best_B
        best_fitting_B = cv2.erode(best_fitting_B, k, iterations=2)

    return model_best_B, best_fitting_B


def match_muscleC(frame, C, seg_C, v_range=range(-32, 0, 4), h_range=range(-20, 6, 4), t_range=range(-6, 10, 2)):
    max_fitting_area_C = 0
    model_best_C = None
    best_fitting_C = None

    y, x, _ = frame.shape
    for vertical in v_range:
        for horizontal in h_range:
            trans_mat = np.array([[1, 0, horizontal], [0, 1, vertical]], np.float32)
            affine_C = cv2.warpAffine(C, trans_mat, (x, y))

            for theta in t_range:
                rotate_mat = cv2.getRotationMatrix2D((x / 2, y / 2), theta, 1)
                rotate_C = cv2.warpAffine(affine_C, rotate_mat, (x, y))

                fitting_C = cv2.bitwise_and(rotate_C, seg_C)
                fitting_area = np.sum(fitting_C)

                if fitting_area > max_fitting_area_C:
                    max_fitting_area_C = fitting_area
                    model_best_C = rotate_C
                    best_fitting_C = cv2.bitwise_and(rotate_C, frame)

    # 避免該片段模型無法擬合
    if model_best_C is None or max_fitting_area_C < 20000:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        trans_mat = np.array([[1, 0, best_horizontal_A], [0, 1, best_vertical_A]], np.float32)
        affine_C = cv2.warpAffine(C, trans_mat, (x, y))

        rotate_mat = cv2.getRotationMatrix2D((x / 2, y / 2), best_theta_A, 1)
        model_best_C = cv2.warpAffine(affine_C, rotate_mat, (x, y))
        best_fitting_C = model_best_C
        best_fitting_C = cv2.erode(best_fitting_C, k, iterations=2)

    return model_best_C, best_fitting_C

