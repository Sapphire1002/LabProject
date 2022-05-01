import pydicom
import cv2
import os
import time
import logging
import json
import re


video_dcm_dir = 'L:/Lab_Data/dcm Data2/'
video_dcm_dir_list = os.listdir(video_dcm_dir)

# 迭代每個資料夾並讀取裡面的 DCM 檔案
all_list = list()
all_files = list()
error_file = 0

for curr_dir in video_dcm_dir_list:
    curr_path = os.path.join(video_dcm_dir, curr_dir)
    curr_dir_list = os.listdir(curr_path)

    # 儲存每個 curr_dir_list 底下的所有資料夾的路徑
    reg_list = list()
    for dirs in curr_dir_list:
        if os.path.isdir(os.path.join(curr_path, dirs)):
            dir_path = curr_path + "/" + dirs
            all_list.append(dir_path)

for dcm_dir in all_list:
    curr_path = dcm_dir + '/IMG001/'
    dcm_files = os.listdir(curr_path)
    all_files.append(len(dcm_files))

    for dcm_file in dcm_files:
        write_dir = 'E:/MyProgramming/Python/Project/implement/heart recognize/dicom data 2/'
        write_path = write_dir + curr_path[len(video_dcm_dir):]

        if not os.path.isdir(write_path):
            os.makedirs(write_path)
        write_path = write_path + dcm_file
        write_path = write_path.replace(".DCM", ".json")

        dcm_path = curr_path + dcm_file
        dcm = pydicom.dcmread(dcm_path)
        case_dict = dict()
        for key in dcm.keys():
            s = '%s' % dcm[key]
            key = str(key)

            case_dict[key] = s[len(key)+1:]
            # print(case_dict)
            # pattern = r"\D"

            # if key == '(0018, 1088)':
            #     print(case_dict[key])
            #     print(f'{int(re.split(pattern, case_dict[key])[-2])}')

            # if key == '(0018, 0040)':
            #     print(case_dict[key])
            #     print(f'{int(re.split(pattern, case_dict[key])[-2])}')

        with open(write_path, 'a+') as obj:
            json.dump(case_dict, obj)

        # print(dcm)
        # print('ID:', dcm.PatientID)
        # print('Name:', dcm.PatientName)
        # print('Birth:', dcm.PatientBirthDate)
        # year, month, day = dcm.PatientBirthDate[:4], dcm.PatientBirthDate[4:6], dcm.PatientBirthDate[6:]
        # print(year, month, day)
        # print('Sex:', dcm.PatientSex)
        # print('StudyDate:', dcm.StudyDate)
        # print('StudyTime:', dcm.StudyTime)
        # print('InstitutionName:', dcm.InstitutionName)
        # print('Manufacturer:', dcm.Manufacturer)
        # print(write_path)

