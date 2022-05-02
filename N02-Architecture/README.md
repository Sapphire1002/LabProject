## 心臟超音波影像架構圖說明
+ **Dicom Data 架構**  
![Dicom Data](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/Dicom%20Data%20%E6%9E%B6%E6%A7%8B.PNG)  
說明:  醫學超音波影像的 Dicom 資料會有的內容  

+ **Color Doppler 架構 & BreakDown**
![Color Doppler](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/Color%20Doppler%20%E6%9E%B6%E6%A7%8B.jpg)  
說明: 需要先找出影像中白色扇形區域, 接著按照扇形區域內的顏色區分嚴重程度    
![Color Doppler BreakDown](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/Color%20Doppler%20BreakDown.jpg)  

+ **PLA Matching 架構 & BreakDown**  
![PLA Matching](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/PLA%20Matching%20%E6%9E%B6%E6%A7%8B.jpg)    
說明: 利用標準心臟模型和影像每幀做匹配, 區分腔室位置  
![PLA Matching BreakDown](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/PLA%20Matching%20BreakDown.jpg)  

+ **A4C Segmeantation and valve 架構 & BreakDown**  
![A4C Seg. Valve1](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/A4C%20Segmentation%20and%20valve%20%E6%9E%B6%E6%A7%8B.PNG)  
說明: 定義 Apical four chamber view 的腔室名稱以及定義二尖瓣位置  
![A4C Seg. Valve1 BreakDown](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/A4C%20Segmentation%20and%20valve%20BreakDown.PNG)

* v2: 結合了 Multi-Threshold 以及調整定義瓣膜的方式
![A4C Seg. Valve2](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/A4C%20Segmentation%20and%20valve%20%E6%9E%B6%E6%A7%8Bv2.PNG)  
![A4C Seg. Valve2 BreakDown](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/A4C%20Segmentation%20and%20valve%20BreakDown%20v2.PNG)  

+ **Muscle Semantic 架構 & BreakDown**  
![Muscle Semantic](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/Muscle%20Semantic%20%E6%9E%B6%E6%A7%8B%E5%9C%96.PNG)  
說明: 至目前為止的進度的整個系統專案架構. 目標是計算 LVEF & GLS 數值. 底下的 BreakDown 為 Multi-Threshold 後面的綠色區塊  
![Muscle Semantic BreakDown](https://github.com/Sapphire1002/LabProject/blob/main/N02-Architecture/Muscle%20Semantic%20BreakDown.PNG)  
