from integration_v1 import DopplerModel
from integration_v1 import DicomData
import os


def read_file(dirs):
    case_name_list = os.listdir(input_dir)
    case_category_list = list()
    each_case_video_path = list()

    for case_name in case_name_list:
        case_path = os.path.join(dirs, case_name) + '/'

        case_category_list = os.listdir(case_path)
        case_category_all_path = list()

        for ii in range(len(case_category_list)):
            case_category_path = os.path.join(case_path, case_category_list[ii]) + '/'
            case_category_video_list = os.listdir(case_category_path)
            case_category_video_path = list()

            for j in range(len(case_category_video_list)):
                case_category_video_path.append(os.path.join(case_category_path, case_category_video_list[j]))
            case_category_all_path.append(case_category_video_path)
        each_case_video_path.append(case_category_all_path)

    return case_name_list, case_category_list, each_case_video_path


# input_dir = './1st model data input/'
input_dir = './add data/'
case_num, category, case_video_path = read_file(input_dir)

dcm_dir = '../1st data dcm/'
output_dir = './1st model data output/'

# unit_list = [
#     [[15, 15, 15, 15, 15, 15], [15, 15, 15, 15], [15], [15, 15]],
#     [[15, 15, 17, 17, 17, 17], [15, 15, 17, 17, 17], [15], [15, 15, 17]],
#     [[16, 16, 16, 16, 16], [16, 16, 16, 16, 16, 16], [16], [16, 16]],
#     [[15, 15, 15, 15, 15, 15, 15, 15], [15, 15, 15, 15, 15, 15], [15, 15], [16, 15, 15, 15]],
#     [[13, 13, 13, 13, 13, 13, 15, 15, 15, 13], [13, 13, 14, 14, 15, 15, 15, 15, 13], [13, 13, 13, 13], [13, 13, 13, 13,
#                                                                                                         14, 14, 15]],
#     [[15, 15, 15, 15, 15], [15, 15, 15, 15, 15, 15], [15], [15, 15]],
#     [[15, 15, 15, 15], [15, 15, 15, 15, 15, 15, 15], [15], [15, 15, 15]],
#     [[15, 15, 17], [15, 15, 17, 17, 17, 17], [15, 15], [15, 15, 17, 17]],
#     [[13, 13, 13, 17, 15], [13, 17, 17, 17, 17, 17, 17, 15], [13], [13, 17, 17, 17]],
#     [[14, 14, 15, 15, 15, 15, 15, 13], [14, 15, 15, 15, 15, 15, 13], [14, 15, 15], [14, 15, 15, 15, 15, 15, 15]],
#     [[17, 17, 17, 17, 17, 17], [17, 17, 17, 17, 17], [17, 17], [17, 17]],
#     [[13, 13, 15, 15], [13, 13, 15, 15, 15, 15], [13], [13, 15, 15]],
#     [[12, 12, 12, 12, 12, 15, 15, 15, 12], [12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 12], [12, 12], [12, 12, 15, 15, 15,
#                                                                                                     15, 15]],
#     [[16, 16, 16, 17, 16], [16, 16, 17, 17, 17, 17, 17, 17, 16], [16, 16], [16, 16, 16, 17, 17, 17, 17, 17, 17]],
#     [[15, 15, 15, 15, 15], [15, 15, 15, 15, 15], [15], [15, 15, 15, 15, 15, 15]],
#     [[15, 15, 16, 16, 16], [15, 16, 16, 16], [15], [15, 15, 15, 16, 16]],
#     [[17, 17, 17, 17, 17, 17], [17, 17, 17, 17, 17, 17, 17], [17, 17], [17, 17, 17]],
#     [[14, 14, 14, 15, 16, 16, 16, 16, 16], [14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 16], [14, 14], [14, 14, 14, 15, 15,
#                                                                                                     15, 16]],
#     [[13, 13, 13, 15, 17, 15, 13], [13, 13, 15, 15, 17, 15, 15, 13], [13, 13], [13, 13, 15, 15]],
#     [[13, 13, 13, 13], [13, 13, 13, 13, 13], [13], [13, 13, 13, 13, 13, 13]]
# ]

# case name: 90266008
# unit_list = [
#     [[15, 15, 15, 15, 15], [15, 15, 15, 15, 15, 15, 15, 15], [15], [15, 15, 15, 15]]
# ]

# case name: 01608015, 01627562
unit_list = [
    # [[15, 15, 15, 15], [15, 15, 15, 15, 15, 15, 15], [15], [15, 15, 15]],
    [[15, 15, 15, 15, 15, 15, 15], [15, 15, 15, 15, 15], [15], [15, 15, 15, 15]]
]

for case_index in range(len(case_video_path)):
    curr_case_name = case_num[case_index]

    for curr_index in range(len(case_video_path[case_index])):
        curr_pos = category[curr_index]

        for i in range(len(case_video_path[case_index][curr_index])):
            # print('case name:', curr_case_name)
            # print('curr pos:', curr_pos)
            # print('path:', case_video_path[case_index][curr_index][i])
            # print()

            path = case_video_path[case_index][curr_index][i]
            dcm_path = dcm_dir + path.split('/')[-1].replace('.avi', '.DCM')

            doppler = DopplerModel(path, diagnosisPos=curr_pos, case_name=curr_case_name)
            DCM = DicomData(dcm_path)

            mask_region = doppler.find_region()
            doppler.standard_unit(unit_list[case_index][curr_index][i])

            curr_res = list()
            while doppler.ret:
                frame = doppler.get_frame()

                if not doppler.ret:
                    break

                original = frame.copy()
                frame[mask_region != 255] = [0, 0, 0]
                mask_connect = doppler.color_info(frame)
                result = doppler.show(original, mask_connect, 0.1, True, dcm_path)
                curr_res.append(result)

                # 分級嚴重程度
                # img_path = output_dir + curr_case_name + '/' + curr_pos + '/' + path.split('/')[-1]
                # img_path = img_path.replace('.avi', '.png')

            curr_res = doppler.show_TimeBar(curr_res)

            write_path = output_dir + curr_case_name + '/' + curr_pos + '/' + path.split('/')[-1]
            print(write_path)
            doppler.writeVideo(curr_res, write_path, fps=20)
