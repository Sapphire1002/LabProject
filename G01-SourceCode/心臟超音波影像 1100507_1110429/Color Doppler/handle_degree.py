from integration_v1 import DopplerModel
import os
import numpy as np
import matplotlib as mpt
import matplotlib.pyplot as plt


def read_file(dirs):
    position_list = os.listdir(dirs)
    each_pos_path_list = list()

    for pos in position_list:
        each_pos_path = os.path.join(dirs, pos) + '/'
        each_pos_path_list.append(each_pos_path)

    each_pos_video_path_list = list()
    for video_path in each_pos_path_list:
        video_list = os.listdir(video_path)
        curr_video_path_list = list()

        for curr_video in video_list:
            curr_video_path = os.path.join(video_path, curr_video)
            curr_video_path_list.append(curr_video_path)
        each_pos_video_path_list.append(curr_video_path_list)

    return each_pos_video_path_list, position_list


input_dir = './1st model data/'
total_path, pos_list = read_file(input_dir)

unit_list = [
    [
        15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 16, 16, 16,
        16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 13, 13, 13, 13, 13,
        13, 15, 15, 15, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 17, 13, 13, 13, 17, 15, 14, 14, 15, 15, 15, 15, 15, 13,
        17, 17, 17, 17, 17, 17, 13, 13, 15, 15, 12, 12, 12, 12, 12,
        15, 15, 15, 12, 16, 16, 16, 17, 16, 15, 15, 15, 15, 15, 15,
        15, 16, 16, 16, 17, 17, 17, 17, 17, 17, 14, 14, 14, 15, 16,
        16, 16, 16, 16, 13, 13, 13, 15, 17, 15, 13, 13, 13, 13, 13
    ],

    [
        15, 15, 15, 15, 15, 15, 17, 17, 17, 16, 16, 16, 16, 16, 16,
        15, 15, 15, 15, 15, 15, 13, 13, 14, 14, 15, 15, 15, 15, 13,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        17, 17, 17, 17, 13, 17, 17, 17, 17, 17, 17, 15, 14, 15, 15,
        15, 15, 15, 13, 17, 17, 17, 17, 17, 13, 13, 15, 15, 15, 15,
        12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 12, 16, 16, 17, 17,
        17, 17, 17, 17, 16, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17,
        17, 17, 17, 17, 17, 17, 14, 14, 14, 15, 15, 15, 16, 16, 16,
        16, 16, 13, 13, 15, 15, 17, 15, 15, 13, 13, 13, 13, 13, 13
    ],

    [
        15, 15, 16, 15, 15, 13, 13, 13, 13, 15, 15, 15, 15, 13, 14,
        15, 15, 17, 17, 13, 12, 12, 16, 16, 15, 15, 17, 17, 14, 14,
        13, 13, 13
    ],

    [
        15, 15, 15, 15, 17, 16, 16, 16, 15, 15, 15, 13, 13, 13, 13,
        14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 13, 17, 17,
        17, 14, 15, 15, 15, 15, 15, 15, 17, 17, 13, 15, 15, 12, 12,
        15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 17, 17, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17, 14, 14,
        14, 15, 15, 15, 16, 13, 13, 15, 15, 13, 13, 13, 13, 13, 13
    ]
]

txt_dir = './1st model data/'

# for category_index in range(len(total_path)):
#     for curr_index in range(len(total_path[category_index])):
#         doppler = DopplerModel(total_path[category_index][curr_index])
#         mask_region = doppler.find_region()
#         doppler.standard_unit(unit_list[category_index][curr_index])
#
#         while doppler.ret:
#             frame = doppler.get_frame()
#
#             if not doppler.ret:
#                 break
#
#             original = frame.copy()
#             frame[mask_region != 255] = [0, 0, 0]
#             mask_connect = doppler.color_info(frame)
#             result = doppler.show(original, mask_connect, 0.1)
#         txt_path = txt_dir + pos_name[category_index] + '/' + 'AreaInfo.txt'
#         doppler.gen_txt(txt_path=txt_path)

quartile_list = list()
for pos_name in pos_list:
    file_area_info = './1st model data/' + pos_name + '/AreaInfo.txt'
    f = open(file_area_info, 'r')
    total_list = list()

    for video_info in f.readlines():
        if len(video_info) > 4:
            area_info = video_info[1:len(video_info)-2].split(', ')
            for i in range(len(area_info)):
                area_info[i] = float(area_info[i])
                total_list.append(area_info[i])
    total_list = np.asarray(total_list)
    x, y = np.unique(total_list, return_counts=True)
    x = x[1:len(x)-1]
    quartile = np.percentile(x, [25, 50, 75])
    quartile_list.append(quartile)

    # plt.figure(1)
    # plt.title(pos_name)
    # plt.xlabel('Area(sq.cm)')
    # plt.ylabel('count')
    # plt.scatter(x, y, c='b', s=10)
    # plt.savefig('./1st model data/' + pos_name + '/visual.png')
    # plt.clf()
    # print('Pos:', pos_name)
    # print(np.unique(total_list, return_counts=True))

print(quartile_list)
