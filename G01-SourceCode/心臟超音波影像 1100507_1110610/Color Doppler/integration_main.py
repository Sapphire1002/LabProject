from integration_v1 import DopplerModel
from integration_v1 import DicomData
import glob


def read_file(video_dir, name='*.avi'):
    all_video_path = glob.glob(video_dir + name)
    return all_video_path


dcm_file_path = read_file('..\\1st data dcm\\', name='*.DCM')
video_path = read_file('..\\1st data all\\')
write_dir = './1st data test1/'
text_path = './1st data test1.txt'

unit_list = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17, 17,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    13, 13, 13, 13, 13, 13, 14, 14, 15, 15, 15, 15, 13,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 17, 17, 17,
    13, 13, 13, 13, 17, 17, 17, 17, 17, 17, 15,
    14, 14, 15, 15, 15, 15, 15, 15, 15, 13,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15,
    12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 12,
    16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 16,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16, 16,
    13, 13, 13, 13, 15, 15, 17, 15, 15, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
]

i = 0
for path in video_path:
    file_name = path.split('\\')[-1]
    case_name = file_name.split('_')[0]

    # Doppler 和 DCM class
    doppler = DopplerModel(path)
    DCM = DicomData(dcm_file_path[i])

    # 1. 找出 Doppler 有效扇形區域
    mask_region = doppler.find_region()

    # 2. 輸入標準長度
    doppler.standard_unit(unit_list[i])

    curr_res = list()
    while doppler.ret:
        frame = doppler.get_frame()

        if not doppler.ret:
            break

        original = frame.copy()

        frame[mask_region != 255] = [0, 0, 0]
        # 3. 找出有問題的顏色區域
        mask_connect = doppler.color_info(frame)

        # if mask_connect is None:
        #     continue

        # 4. 展示 Doppler 結果
        result = doppler.show(original, mask_connect, 0.1, True, dcm_path=dcm_file_path[i])
        curr_res.append(result)

    # 5. 產生 txt 檔案後 判斷嚴重程度
    # doppler.gen_txt(text_path)

    # 6. 顯示時間條
    curr_res = doppler.show_TimeBar(curr_res)

    print(i)
    # Last. 將影片寫入
    write_path = write_dir + file_name
    doppler.writeVideo(curr_res, write_path, fps=20)
    i += 1
