import os
import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json


def AllFiles(InputDir, extensionName='avi'):
    targetList = list()
    ext = len(extensionName)

    for root, dirs, files in os.walk(InputDir):
        for f in files:
            if f[-ext:].lower() == extensionName:
                targetList.append(os.path.join(root, f))

    return targetList


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
    gray[gray <= 20] = 0

    return gray.ravel()


# 建立資料集和標籤
path = 'L:\\Classification\\Avg Image\\'
dir_list = os.listdir(path)

labels = list()
datasets = list()

trainData = list()
testData = list()

for index in range(0, len(dir_list)):
    curr_dir = os.listdir(path + dir_list[index])

    # 將訓練資料拆分成各類總數的 0.2 相加
    testData.append(list())
    samples = [k for k in range(len(curr_dir))]
    training = random.sample(samples, int(len(samples) * 0.2))
    for i in range(len(samples)):
        if i not in training:
            testData[index].append(i)

    for ind, file in enumerate(curr_dir):
        if ind not in training:
            continue

        img_path = path + '/' + dir_list[index] + '/' + file
        ori = cv2.imread(img_path)
        data = adjust_img(ori, 0.1)
        datasets.append(data)
        labels.append(index + 1)

    print('該類別的資料數量: ', len(curr_dir))
    print('目前資料集的資料量: ', len(datasets))
    print("當前類別標籤: %d " % (index + 1))

datasets = np.array(datasets, np.float32)
labels = np.array(labels, np.int32)

print('dataset shape: ', datasets.shape)
print('labels shape: ', labels.shape)

# 訓練模型(採用 cv2 方式)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.1)
svm.setGamma(1.0)

st = time.time()
svm.trainAuto(datasets, cv2.ml.ROW_SAMPLE, labels)
end = time.time()

print('訓練模型時間: ', round(end - st, 2), '秒')
print('使用模型類型: ', svm.getType())
print('模型核心類型: ', svm.getKernelType())
print('模型參數: ')
print('C: ', svm.getC())
print('Gamma: ', svm.getGamma())

model_path = './svm_model_Thres10.xml'
svm.save(model_path)


# Use Model
model = cv2.ml.SVM_load("./svm_model_Thres10.xml")
target_dir = 'L:\\Classification\\Avg Image\\'
all_dir = os.listdir(target_dir)

for index in range(0, len(all_dir)):
    curr_dir = os.listdir(target_dir + all_dir[index])

    for ind, file in enumerate(curr_dir):
        # Inside + Outside
        # if ind not in testData[index]:
        #     continue

        img_path = target_dir + '/' + dir_list[index] + '/' + file
        print(img_path)

        ori = cv2.imread(img_path)
        data = adjust_img(ori, 0.1)
        feature = np.array([data], np.float32)
        label = model.predict(feature)[1][0][0]

        write_path = 'L:/Predict/000%d' % label + '/' + file
        print('current writing path: ', write_path)
        cv2.imwrite(write_path, ori)
    pass

# Confusion Matrix
# 讀取正確資料的資料夾
true_data_path = 'L:\\Classification\\Avg Image\\'
true_data_dir = os.listdir(true_data_path)
curr_total_data = list()

# 建立一個字典 資料為 {filename: label}
data_labels = dict()
for index in range(0, len(true_data_dir)):
    dir_path = os.path.join(true_data_path, true_data_dir[index]) + '/'
    dirs = os.listdir(dir_path)

    curr_total_data.append(len(testData[index]))

    for i, file in enumerate(dirs):
        # if i not in testData[index]:
        #     continue

        data_labels[file] = index + 1

# 讀取預測資料的資料夾
pred_path = 'L:\\Predict\\'
pred_data_dir = os.listdir(pred_path)

# 計算混淆矩陣
confusion_matrix = np.zeros((9, 9), np.int32)

for index in range(0, len(pred_data_dir)):
    dir_path = os.path.join(pred_path, pred_data_dir[index]) + '/'
    dirs = os.listdir(dir_path)

    for file in dirs:
        true_labels = data_labels[file]
        confusion_matrix[true_labels - 1, index] += 1

result = dict()
all_tp = 0
for c in range(0, len(confusion_matrix)):
    tp_fp = confusion_matrix[:, c]
    tp_fn = confusion_matrix[c, :]

    all_tp += confusion_matrix[c, c]
    accuracy = confusion_matrix[c, c] / curr_total_data[c]
    precision = confusion_matrix[c, c] / np.sum(tp_fp)
    recall = confusion_matrix[c, c] / np.sum(tp_fn)
    f1_score = 2 / (1 / precision + 1 / recall)

    result[str(c+1)] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1Score": f1_score}

model_acc = all_tp / sum(curr_total_data)

with open('./SVM Score In 20 Out 80_3.json', 'w+') as j:
    json.dump(result, j)

print('每個類別的資料: ', curr_total_data)
print("Result: \n", result)
print("模型正確率: ", model_acc)

print("X 軸為預測標籤")
print("Y 軸為實際標籤")
print(confusion_matrix)

sns.set()
sns.heatmap(confusion_matrix, square=True, annot=True, cbar=True, fmt='d')
plt.title('confusion matrix')
plt.xlabel('predict value')
plt.ylabel('true value')
plt.show()
