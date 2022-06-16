import FileIO
import pydicom
import cv2
import os


class DCMInit(object):
    """
    class name:
        DCMInit(FilePath, OutputDir=None):
        DCM 轉 avi 或 png，並讀取病人資訊

    parameters:
        FilePath:  輸入當前檔案的絕對路徑
        OutputDir: 輸出 AVI or png 的路徑位置
    """

    def __init__(self, FilePath, OutputDir):
        """
        initialization attributes:

        self.DCMFilePath: DCM 檔案的絕對路徑, str
        self.OutputDir: 輸出資料夾的位置, str
        self.OutputDirPath: 存放輸出資料夾存在的檔案路徑, list
        """

        self.DCMFilePath = FilePath
        self.OutputDir = OutputDir
        self.OutputDirPath = FileIO.AllFiles(OutputDir)

        if not os.path.isdir(self.OutputDir):
            os.makedirs(self.OutputDir)

        CheckAVI = FilePath.split("\\")[-1].replace('DCM', 'avi')
        CheckAVI = self.OutputDir + CheckAVI

        if CheckAVI in self.OutputDirPath:
            print(f'{FilePath}, 該 DCM 檔案已轉成 avi or png 在輸出資料夾裡. (可忽略此訊息)')
        else:
            self._ConvDCMToAVI()

        self.AVIPath = CheckAVI

    def _ConvDCMToAVI(self):
        """
        method name:
            _ConvDCMToAVI():
            將 DCM 檔轉成 avi 或 png 檔
        """

        CurrDCMPath = self.DCMFilePath
        DCMFileName = CurrDCMPath.split('\\')[-1]

        print(f'----- 正在進行 {CurrDCMPath} 轉檔 -----')
        OutputConvPath = self.OutputDir + DCMFileName

        DCMFile = pydicom.dcmread(CurrDCMPath)

        try:
            DCMData = DCMFile.pixel_array.shape

        except AttributeError:
            raise AttributeError(f'AttributeError: {CurrDCMPath} 該路徑的 DCM 檔無法被正確讀取')

        if len(DCMData) > 3:
            frames, y, x, channel = DCMData

            OutputConvPath = OutputConvPath.replace('.DCM', '.avi')
            video_write = cv2.VideoWriter(OutputConvPath, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))

            for i in range(frames):
                bgr = cv2.cvtColor(DCMFile.pixel_array[i], cv2.COLOR_YUV2BGR)
                res = cv2.resize(bgr, (x, y))
                video_write.write(res)
            video_write.release()

        elif len(DCMData) == 3:
            if DCMData[-1] == 3:
                OutputConvPath = OutputConvPath.replace('.DCM', '.png')
                r, g, b = cv2.split(DCMFile.pixel_array)
                merge = cv2.merge([b, g, r])
                cv2.imwrite(OutputConvPath, merge)

            else:
                frames, y, x = DCMData

                OutputConvPath = OutputConvPath.replace('.DCM', '.avi')
                video_write = cv2.VideoWriter(OutputConvPath, cv2.VideoWriter_fourcc(*'MJPG'), 30, (x, y))

                for i in range(frames):
                    res = cv2.resize(DCMFile.pixel_array[i], (x, y))
                    video_write.write(res)
                video_write.release()

        else:
            OutputConvPath = OutputConvPath.replace('.DCM', '.png')
            cv2.imwrite(OutputConvPath, DCMFile.pixel_array)

