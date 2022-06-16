import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json


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


# 讀取正確資料的資料夾
true_data_path = 'L:\\Classification\\Avg Image\\'
true_data_dir = os.listdir(true_data_path)
curr_total_data = list()

# 建立一個字典 資料為 {filename: label}
data_labels = dict()
for index in range(0, len(true_data_dir)):
    dir_path = os.path.join(true_data_path, true_data_dir[index]) + '/'
    dirs = os.listdir(dir_path)
    curr_total_data.append(len(dirs))

    for file in dirs:
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

with open('./SVM Score_v2.json', 'w+') as j:
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


