from ProcessesCtrl import Process

InputDCMDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\Test DCM\\"
OutputAVIDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\Test AVI\\"
OutputSkeletonizeDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\Test Skeleton\\"
OutputSegmentDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\Test Segment\\"
OutputGLSDir = "E:\\MyProgramming\\Python\\Project\\implement\\heart recognize\\System2\\Test GLS\\"

Process(InputDCMDir, OutputAVIDir, OutputSkeletonizeDir, OutputSegmentDir, OutputGLSDir)
