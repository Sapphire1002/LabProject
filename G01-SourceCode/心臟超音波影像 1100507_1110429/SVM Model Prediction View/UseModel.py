import cv2
import os
import numpy as np


def adjust_img(img, scale=0.1):
    """
    function(img[, scale=0.1]):
        按照 scale 調整原始圖片的比例, 轉成灰階後展開為一維陣列

    return:
        gray.ravel()
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = gray.shape
    shape = (int(x * scale), int(y * scale))
    gray = cv2.resize(gray, shape, cv2.INTER_AREA)
    return gray.ravel()


def AllFiles(InputDir, extensionName='avi'):
    targetList = list()
    ext = len(extensionName)

    for root, dirs, files in os.walk(InputDir):
        for f in files:
            if f[-ext:].lower() == extensionName:
                targetList.append(os.path.join(root, f))

    return targetList


model = cv2.ml.SVM_load("./svm_model.xml")
target_dir = 'L:\\Classification\\Avg Image\\'
all_img = AllFiles(target_dir, 'png')

for file in all_img:
    FileName = str(file.split("\\")[-1])

    ori = cv2.imread(file)
    data = adjust_img(ori, 0.1)
    feature = np.array([data], np.float32)
    label = model.predict(feature)[1][0][0]

    write_path = 'L:/Predict/000%d' % label + '/' + FileName
    print('current writing path: ', write_path)
    cv2.imwrite(write_path, ori)
    pass
