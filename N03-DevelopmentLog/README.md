## PPT 檔案連結
[VHDL & FPGA](https://docs.google.com/presentation/d/1_oMd8nB5ge3vATgKx2VlhOD2gKH16Xzx/edit?usp=sharing&ouid=114732633741530754400&rtpof=true&sd=true)  
[心臟影像](https://docs.google.com/presentation/d/1t9eHXb45PI_M94EGgWX3N9IUr5dVp5rb/edit?usp=sharing&ouid=114732633741530754400&rtpof=true&sd=true)  


## VHDL & FPGA
#### **每周進度**
<details>
  <summary> Week 1 環境建置 </summary>
  日期: 2020.10.27 - 2020.10.30  
  
  專案資料夾: [00 pre_test](https://github.com/Sapphire1002/VHDL/tree/main/00%20pre_test "專案連結")  
  進度:  
  建置 Vivado 環境  
  查詢 VHDL 語法及資料  
  
</details>

<details>
  <summary> Week 2 VGA 螢幕掃描 </summary>
  日期: 2020.10.30 - 2020.11.06  
  
  專案資料夾: [01 video_out_screen_scan](https://github.com/Sapphire1002/VHDL/tree/main/01%20video_out_screen_scan "專案連結")  
  進度:  
  查詢 VHDL 語法及資料  
  了解螢幕掃描時間及程式設計流程  
  了解螢幕輸出RGB時的原理  
  完成螢幕掃描  
  
<details>
  <summary> 實作部分 </summary>
  
  * 了解螢幕掃描時間及程式設計流程  
  ![螢幕掃描流程圖](https://github.com/Sapphire1002/VHDL/blob/main/01%20video_out_screen_scan/%E8%9E%A2%E5%B9%95%E6%8E%83%E6%8F%8F%E6%B5%81%E7%A8%8B%E5%9C%96.PNG)  
  * 原本螢幕畫面  
  ![原本螢幕畫面](https://github.com/Sapphire1002/VHDL/blob/main/01%20video_out_screen_scan/1106_ori.jpg)  
  * 掃描後的螢幕畫面  
  ![掃描後的螢幕畫面](https://github.com/Sapphire1002/VHDL/blob/main/01%20video_out_screen_scan/1106_result.jpg)  
</details>  

<details>
  <summary> 問題討論 </summary>
  
  ![Q](https://github.com/Sapphire1002/VHDL/blob/main/01%20video_out_screen_scan/1106_q1.PNG)  
  - [x] 已解決  
        解決方式: 在 \*.xdc 檔案時脈的程式碼要加上 IOSTANDARD 並給電壓 LVCMOS33  
  - [ ] 未解決
</details> 
</details>

<details>
  <summary> Week 3 期中考 </summary>
  期中考週
</details>

<details>
  <summary> Week 4 VGA 螢幕上顯示圖形 </summary>
  日期: 2020.11.13 - 2020.11.20 
  
  專案資料夾: [02 video_out_graphics_move](https://github.com/Sapphire1002/VHDL/tree/main/02%20video_out_graphics_move "專案連結")  
  進度:  
  在 VGA 螢幕上顯示正方形、圓形、三角形  
  使螢幕上的圖形移動  
  
<details>
  <summary> 實作部分 </summary>
  
  * 顯示圖形  
  ![顯示圖形](https://github.com/Sapphire1002/VHDL/blob/main/02%20video_out_graphics_move/1120_Video_out_%E5%9C%96%E5%BD%A2.jpg)  
  [圖形移動影片](https://drive.google.com/file/d/1x19yr52etBxJ1drvSTe1m-OdFJPInAqK/view?usp=sharing)  
</details>

<details>
  <summary> 問題討論 </summary>  
  
  ![Q](https://github.com/Sapphire1002/VHDL/blob/main/02%20video_out_graphics_move/1120_video_out_que01.png)  
  - [x] 已解決  
        解決方式: 重新建立一個專案    
  - [ ] 未解決  
  * 三角形在一開始的地方會有問題  
  - [x] 已解決  
        解決方式: 利用數學的線性規劃來判斷點位於直線方程式哪邊      
  - [ ] 未解決   
  * 兩個 process() 傳值的方法  
  - [x] 已解決  
        解決方式:  
            1\. 宣告一個 signal, 類型為 std_logic_vector  
            2\. 在第二個 process 寫一個區域變數(variable)來接收傳入的值  
            3\. 在第二個 process 賦值給 第一步驟宣告的 signal  
            4\. 在第一個 process 接收值, 若要轉成十進制則使用(conv_integer(variable, bits))  
            `conv_integer() 需要有 ieee.std_logic_arith.all 檔案`  
  - [ ] 未解決  
</details>  
</details>

<details>
  <summary> Week 5 VGA 螢幕玩乒乓球遊戲 </summary>
  日期: 2020.11.20 - 2020.11.27  
  
  專案資料夾: [03 video_out_pingpong_vga](https://github.com/Sapphire1002/VHDL/tree/main/03%20video_out_pingpong_vga "專案連結")  
  進度:  
  使用 VGA 螢幕顯示且玩乒乓球遊戲  
  依據打擊的位置球往不同的方向飛   
  
<details>
  <summary> 實作部分 </summary>
  
  [乒乓球實作影片1](https://drive.google.com/file/d/1cx5e87o8t2VbzjyqEA-TgOCNKX9wB-Pk/view?usp=sharing)    
  [乒乓球實作影片2](https://drive.google.com/file/d/1H7-WLFPHP_LOq9tE38c5P5waZKvh8pJ7/view?usp=sharing)  
</details>

<details>
  <summary> 問題討論 </summary> 
  
  * 兩邊的檔板若超出邊界會直接消失並從另一端出現 
  - [ ] 已解決        
  - [x] 未解決  
</details>
</details>

<details>
  <summary> Week 6 LED/七段計數器 </summary>
  日期: 2020.11.27 - 2020.12.04  
  
  專案資料夾: [04 counter](https://github.com/Sapphire1002/VHDL/tree/main/04%20counter "專案連結")  
  進度:  
  計數器 0 ~ 9， 9 ~ 0  
  讓兩個計數器可自由設定上下限  
  計數的結果顯示在 LED 及 七段顯示器上  
  
<details>
  <summary> 實作部分 </summary>
  
  * 上數波形模擬    
  ![上數波形模擬](https://github.com/Sapphire1002/VHDL/blob/main/04%20counter/%E4%B8%8A%E6%95%B8%E8%A8%88%E6%95%B8%E5%99%A8(0_9%E6%B3%A2%E5%BD%A2).PNG)  
  * 下數波形模擬  
  ![下數波形模擬](https://github.com/Sapphire1002/VHDL/blob/main/04%20counter/%E4%B8%8B%E6%95%B8%E8%A8%88%E6%95%B8%E5%99%A8(9_0%20%E6%B3%A2%E5%BD%A2).PNG)  
  * 自定義計數器波形模擬  
  ![自定義計數器波形](https://github.com/Sapphire1002/VHDL/blob/main/04%20counter/%E8%87%AA%E5%AE%9A%E7%BE%A9%E8%A8%88%E6%95%B8%E5%99%A8(%E6%B3%A2%E5%BD%A2).PNG)  

  [LED 上數影片](https://drive.google.com/file/d/1h8_54hwukTBwddUCOMGQsIpPvyr5TOIP/view?usp=sharing)  
  [LED 下數影片](https://drive.google.com/file/d/1HvNs_3RmeN6pVpBwUH8IC6rxIaLaB1HN/view?usp=sharing)  
  影片說明:  
  影片中的 LED 最左邊為 8，最右邊為 1。 數字 9 則顯示 8 和 1，也就是會同時亮最左邊和最右邊
</details>

<details>
  <summary> 問題討論 </summary> 
  
  * 七段顯示器尚未研究怎麼使用
  - [x] 已解決  
        解決方式: FPGA 板子上的七段顯示器無法使用, 使用外接七段顯示器來處理        
  - [ ] 未解決 
</details>
</details>
  
<details> 
  <summary> Week 7 Pulse-width modulation(PWM) </summary>
  日期: 2020.12.04 - 2020.12.11   
  
  專案資料夾: [05 PWM](https://github.com/Sapphire1002/VHDL/tree/main/05%20PWM "專案連結")   
  進度:  
  設計 PWM  
  使用指撥開關設定邊界，並且用有限狀態機來控制兩個計數器的計數。 
  在第一個計數器數的時候 PWM 值為 1，另一個計數器數時值為 0 。  
  最後將結果接上七段顯示器呈現。 
  
<details>
  <summary> 實作部分 </summary>
  
  * PWM 設計流程圖  
  ![PWM 設計流程圖](https://github.com/Sapphire1002/VHDL/blob/main/05%20PWM/PWM_Design_pic.jpg)  
  流程圖說明  
  方框: FPGA 電路  
  箭頭: 輸出訊號  
  菱形: 實際電路  

  * 接上共陽極七段顯示器及 LED 來觀測結果  
  [PWM 接上實際電路觀測結果](https://drive.google.com/file/d/10p-wDH7d7CSU7vLBOSTrHcUxHDYnIQqi/view?usp=sharing)  
  影片說明:  
  LED 代表 PWM 的輸出，紅燈代表上數，黃燈代表下數。
  另外使用 FPGA 板子上的指撥開關來控制邊界。  
  `影片一開始設定 0110，最後設定 0010 `
</details> 
</details>
  
<details>
  <summary> Week 8 LED 乒乓球遊戲 </summary>
  日期: 2020.12.11 - 2020.12.18  
  
  專案資料夾: [06 pingpong_led](https://github.com/Sapphire1002/VHDL/tree/main/06%20pingpong_led "專案連結")  
  進度:  
  設計 LED 乒乓球遊戲    
  使用 LED 當成球在移位，以及兩個按鈕當成 PL1 & PL2，只要達到  
  一邊任意端點就必須在 1個 CLK 內按下該側按鈕。  
  若提早按或者太晚按都算失分，得分時發球權不變，反之換發。  
  最後比分結果由七段顯示器顯示。 
  
<details>
  <summary> 實作部分 </summary>
  
  * 設計 LED 乒乓球遊戲流程圖  
  ![LED 乒乓球遊戲流程圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/pingpong_programming_pic.jpg)  
  * LED 乒乓球遊戲 VHDL 狀態圖    
  ![LED 乒乓球遊戲狀態圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/pingpong_led_pic.jpg)   
  狀態圖說明:    
  000: PL1 發球前的狀態  
  001: PL2 發球前的狀態  
  010: LED 右移  
  011: LED 左移  
  100: PL1 接到球  
  101: PL2 接到球  
  110: PL1 當前分數  
  111: PL2 當前分數  
  btn1, btn2: 代表 PL1, PL2  `電路為正邏輯`  
  pos: 球的當前位置  

  * 接上實際電路觀測結果  
  [實際電路觀測結果](https://drive.google.com/file/d/17KoJ02tQW8P4xKnkNdryfAqvog-4ffQe/view?usp=sharing)   
  影片說明:  
  左邊的按鈕為 PL1， 右邊的按鈕為 PL2，左邊的七段為 PL1 分數，右邊的七段為 PL2 分數。
</details>

<details>
  <summary> 問題討論 </summary>  
  
  * 目前 LED 的部分不會移動，但是計分判斷和按鈕控制流程是正常功能  
  - [ ] 已解決        
  - [x] 未解決   
</details>
</details> 

<details>
  <summary> Week 9 專案管理 </summary>
  日期: 2020.12.18 - 2020.12.25  
  
  處理 GitHub 專案管理  
  [操作連結](https://drive.google.com/file/d/1kbkaADANnAS-PVTFHqxI0UQdvAd30b4R/view?usp=sharing "PPT連結")  
  
</details>

<details>
  <summary> Week 10 LED 乒乓球遊戲 </summary>
  日期: 2020.12.25 - 2021.01.01  
  
  專案資料夾: [06 pingpong_led](https://github.com/Sapphire1002/VHDL/tree/main/06%20pingpong_led "專案連結")  
  進度:  
  修正 LED 不會移動的問題  
  重新設計流程圖和狀態圖  
  完成 LED 乒乓球遊戲  
  
<details>
  <summary> 實作部分 </summary>
  
  * 設計 LED 乒乓球遊戲流程圖  
  ![LED 乒乓球遊戲流程圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/pingpong_programming_pic_v2.jpg)  
  * LED 乒乓球遊戲 Mealy 狀態圖 & FPGA 電路圖      
  ![LED 乒乓球遊戲狀態圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/pingpong_led_pic_v2.jpg)       
  電路&參數說明:  
  btn1: 玩家1  
  btn2: 玩家2  
  MealyFSM: 米利型有限狀態機  
  PL1_score: 玩家1 分數  
  PL2_score: 玩家2 分數  
  cnt: LED 移動的當前位置  
  freq_div: 除頻  
  serve: 控制發球權  
  狀態說明:  
  s0: 玩家發球前  
  s1: LED右移&PL2是否接到球  
  s2: LED左移&PL1是否接到球  

  * LED 乒乓球遊戲實際遊玩影片   
  [實際遊玩影片](https://drive.google.com/file/d/1XFI0Tmmhyu-u4TRTxHXLS94yamRKo8X2/view?usp=sharing)   
  影片說明:  
  左邊的按鈕為 PL1，右邊的按鈕為 PL2，上面的七段為 PL1 分數，下面的七段為 PL2 分數。
</details>  

<details>
  <summary> 問題討論 </summary>   
  
  * 之前問題  
  * 目前 LED 的部分不會移動，但是計分判斷和按鈕控制流程是正常功能  
  - [x] 已解決  
        解決方式: 重新設計狀態圖和流程圖來處理本項問題  
  - [ ] 未解決   
  * Vivado 會無法偵測到 FPGA 板子的問題  
  - [x] 已解決  
        解決方式: 到對應版本的vivado資料夾目錄下找到 install_digilent.exe 並執行  
        `例如: D:\Vivado\2019.2\data\xicom\cable_drivers\nt64\digilent\install_digilent.exe`
  - [ ] 未解決 
</details>
</details>

<details>
  <summary> Week 11 LED 乒乓球遊戲 </summary>
  日期: 2021.01.01 - 2021.01.08  
  
  專案資料夾: [06 pingpong_led](https://github.com/Sapphire1002/VHDL/tree/main/06%20pingpong_led "專案連結")  
  進度:  
  了解 LFSR  
  LED 乒乓球可以有速度的變化  
  
<details>
  <summary> 實作部分 </summary>
  
  * LFSR 原理  
  線性反饋移位暫存器(Linear Feedback Shift Register)  
  給予一個初始值，接著取 n 個位元做 XOR 並將產生的值做為輸入到 MSB 或 LSB，讓暫存器產生移位的效果。  
  作法:  
  ![LFSR 電路圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/lfsr_pingpong_use.jpg)  
  說明:  
  採取 X2 XOR X1 輸入到第一級的 D型正反器。  
  
  * LFSR 實作和測試  
  測試圖:  
  ![LFSR 模擬](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/LFSR_test_result.PNG)  
  說明:  
  程式裡有用一個 temp 來儲存 X2 XOR X1 的值，然而初始值設定為 001、temp 為 0。  
  因此下一次的輸出會受到上一個的temp影響。  
  例如:  
  (X2X1X0, temp): (001, 0) -> (010, 0) -> (100, 1) -> (001, 1) -> (011, 0) -> (110, 1) -> (101, 0)...
  
  * LED 乒乓球遊戲實際遊玩影片   
  [實際遊玩影片](https://drive.google.com/file/d/13V1_zYj_vKg3D8IJxIxA7z4eNMOWi35x/view?usp=sharing)   
  影片說明:  
  有來回打的流程在 4s ~ 11s  
  
</details>  

</details>


<details>
  <summary> Week 12 期末考 </summary>
  期末考週
</details>

<details>
  <summary> Week 13 LED 乒乓球遊戲 </summary>
    日期: 2021.01.15 - 2021.01.22  
  
  專案資料夾: [06 pingpong_led](https://github.com/Sapphire1002/VHDL/tree/main/06%20pingpong_led "專案連結")  
  進度:    
  LED 乒乓球可以有速度的變化  
  
<details>
  <summary> 實作部分 </summary>
  
  * 設計構想流程  
  ![設計構想流程](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/20210119_%E4%B9%92%E4%B9%93%E7%90%83%E8%A8%AD%E8%A8%88%E6%A7%8B%E6%83%B3%E6%B5%81%E7%A8%8B.PNG)    
  
  * 設計構想圖  
  ![設計構想圖](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/20210119_%E4%B9%92%E4%B9%93%E7%90%83%E8%A8%AD%E8%A8%88%E6%A7%8B%E6%83%B3%E5%9C%96.PNG)  
  ![ctrl_ball_clk](https://github.com/Sapphire1002/VHDL/blob/main/06%20pingpong_led/20210119_ctrl_ball_clk%E5%9C%96.PNG)  
  
  說明:  
  clk: 為 FPGA 100MHz 最大速度  
  LFSR_random: 產生 3bits 亂數值，賦值給 Qt  
  freq: 將隨機數的值賦給球速的時間  
  random_value: 依時間把值給 times  
  times: 取 Qt 的最後兩個位元
  clk_div: 球速的最大值  
  ctrl_ball_clk: 依照 times 狀態給予不同的速度值  
  MealyFSM: 01.01的乒乓球進度  

  * LED 乒乓球遊戲實際遊玩影片   
  [實際遊玩影片](https://drive.google.com/file/d/1SCx2BbKd_0MiofaLddfYK3ylky8m_mH7/view?usp=sharing)   
 
  
</details> 
</details>

<details>
  <summary> Week 14 VGA 圖案顯示 </summary>
  日期: 2021.01.21 - 2021.01.27  
  
  專案資料夾: [07 video_out_display_graphics](https://github.com/Sapphire1002/VHDL/tree/main/07%20video_out_display_graphics "專案連結")  
  進度:  
  VGA 顯示 Google 圖案  
  VGA 乒乓球  

<details>
  <summary> 實作部分 </summary>
    <details>
      <summary> IP Catalog 操作 </summary>
      
  * IP Catalog    
  ` 版本: Vivado 2019.2 `  
  RAM & ROM 創建流程    
  ![步驟1](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F1.PNG)  
  ![步驟2](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F2.PNG)  
  ![步驟3](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F2_2.PNG)  
  ![步驟4](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F2_3.PNG)  
  ![步驟5](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F2_4.PNG)  

  * 操作結果  
  ![結果1](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F3.PNG)
  ![結果2](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F3_2.PNG)
  ![結果3](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_IP%E6%AD%A5%E9%A9%9F3_3.PNG)  
        
  </details>
  
   <details>  
     <summary> VGA Display </summary>
      
   * 設計流程
   ![流程圖](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_VGA_display_1.PNG)  
   
   * 實作結果  
   ![Google圖片](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/google_pic_128.png)  
   `size: 128 * 128 `  
   ![顯示](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_VGA_display_2.PNG)
   ![程式](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_VGA_display_2_2.PNG)  
   說明:  
   h_count: 水平當前掃描位置  
   v_count: 垂直當前掃描位置  
   addra: ROM 的地址  
   douta: ROM 在該地址的輸出資料  
   r, g, b: 分別為紅綠藍顏色  

  * VGA PingPong  
  * 設計流程  
  ![流程圖](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_VGA_display_3.PNG)  
  
  * 電路圖  
  ![電路圖](https://github.com/Sapphire1002/VHDL/blob/main/07%20video_out_display_graphics/20210125_VGA_display_4.PNG)  
  說明:  
  紅色箭頭為 外部輸入訊號  
  藍色箭頭為 傳遞參數  
  黃色箭頭為 輸出給外部訊號  
  電路圖說明:  
  clk_divider: 除頻電路  
  clk_div: 除2  
  clk_ball: 除2^21  
  scanner: 處理螢幕掃描及顯示圖形  
  addra: 記憶體位址  
  uut: ROM: 傳遞ROM參數  
  douta: 根據輸出當前addra的資料  
  FSM: 控制遊戲演算及球的移動  
  image_left_x: 圖案左上角座標  
  image_right_y: 圖案右上角座標  
  board_ctrl: 控制板子移動  
  board_left_y: 左側板子的右上角座標  
  board_right_y: 右側板子的左上角座標  

  * 當前實作結果  
  [遊玩影片](https://drive.google.com/file/d/1taIrTT6sPIOCHrO5W4BsGg9jWH7jlPXq/view?usp=sharing)  
  說明:  
  步驟二 圖案移動的地方有狀況，沒辦法顯示完整圖案  
  
   </details>
</details>

<details>
  <summary> 問題討論 </summary>
  
  * Google 圖案移動時會失真  
  - [ ] 已解決   
  - [x] 未解決  
    問題:  
    (目前可能狀況，時序問題)  
    螢幕掃描為 50MHz => 0.02us  
    圖片大小為 128 * 128  
    圖片完全讀取完的時間 327.68us ≒ 0.33ms  

    球移動速度為 0.02us * 2^20 ≒ 20.97ms  
    此時圖片讀取次數 63.55 次  
    球移動時圖片並沒有完整讀取完  

</details>
</details>

<details>
  <summary> Week 15 板子互連乒乓 </summary>
  
  日期: 2021.02.19 - 2021.02.26  
  
  專案資料夾: [08 fpga_connection]("專案連結")  
  進度:  
  兩塊 FPGA 板子互連乒乓  
  
  <details>
  <summary> 實作部分 </summary>
  
  * 設計流程  
  ![流程圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210226_%E8%A8%AD%E8%A8%88%E6%B5%81%E7%A8%8B.PNG)  
  
  * 設計架構圖  
  ![架構圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210226_fpga_connection_%E6%9E%B6%E6%A7%8B%E5%9C%96.PNG)  
  說明:  
  clk: FPGA 100MHz 時脈  
  data: 為 inout 傳輸  
  count: 計算球的位置  
  FSM: 控制球移動的狀態機  
  freq_div: 除頻  
  freq_clk: 除 2^22  
  `目前只有 LED 左移的功能`  
  
  * 當前實作結果  
  [影片連結](https://drive.google.com/file/d/1FJ7SEmzQc0w0w_e17ej9KPOKpSAv0b5X/view?usp=sharing)  
  說明:  
  根據當前的 count 值判斷要傳輸 data資料還是接收資料  
  
  </details>
  
  <details>
  <summary> 問題討論 </summary>
  
   * inout 操作  
   * 一開始使用 reset 另一塊板子的 LED 也會同時移動  
  - [x] 已解決   
        解決方式: 後來採用兩塊板子都有自己的 reset   
  - [ ] 未解決 
  
  </details>

</details>

<details>
  <summary> Week 16 板子互連乒乓 </summary>
  
  日期: 2021.02.26 - 2021.03.04  
  
  專案資料夾: [08 fpga_connection]("專案連結")  
  進度:  
  兩塊 FPGA 板子互連乒乓  

  <details>
  <summary> 實作部分 </summary>
  
  * 設計流程  
  ![流程圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210304_%E8%A8%AD%E8%A8%88%E6%B5%81%E7%A8%8B.PNG)
  
  * 設計架構圖(player_main上, player_other下)    
  ![main架構圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210304_fpga_connect_main_%E6%9E%B6%E6%A7%8B.PNG)  
  說明:  
  ctrl_start: 控制程式開始  
  freq_div: 除頻  
  FSM: 狀態機(目前只有左移)  
  ctrl_stop: 控制停止  
  bit_counter: 計算當前傳送的資料位元  
  data_rw: 控制資料讀寫(目前只有寫) 
  reset_out: 輸出 reset 狀態  
  scl_out: 輸出時序  
  sda: 為 inout 類別負責傳輸資料  
  
  ![other架構圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210304_fpga_connect_other_%E6%9E%B6%E6%A7%8B.PNG)    
  說明:  
  ctrl_start: 控制程式開始  
  freq_div: 除頻  
  FSM: 狀態機(目前只負責更新接收資料)  
  ctrl_stop: 控制停止  
  bit_counter: 計算當前傳送的資料位元  
  data_rw: 控制資料讀寫(目前只有讀)  
  reset_out: 輸出 reset 狀態  
  scl_out: 輸出時序  
  sda: 為 inout 類別負責接收資料  
  receive_reg: 儲存 8bits 的位置  
  
  * 當前實作結果  
  [影片連結](https://drive.google.com/file/d/14m7mvG4YvzyZUhQctgQD8ZyMFNBLdaRd/view?usp=sharing)  
  說明: 
  不確定 8 bits 在接收時的狀態  

  </details>
  
  <details>
  <summary> 問題討論 </summary>
  
   * inout 操作  
   * 8bits資料 在傳送端和接收端沒辦法同步    
  - [x] 已解決   
        解決方式: 後來採用 1 bit 資料傳輸並使用 enable 來控制當前讀寫狀態    
  - [ ] 未解決 
  </details>

</details>

<details>
  <summary> Week 17 板子互連乒乓 </summary>
  
  日期: 2021.03.05 - 2021.03.11  
  
  專案資料夾: [08 fpga_connection]("專案連結")  
  進度:  
  兩塊 FPGA 板子互連乒乓  
  
  <details>
  <summary> 實作部分 </summary>
  
  * 設計流程  
  ![流程圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210311_%E8%A8%AD%E8%A8%88%E6%B5%81%E7%A8%8B.PNG)  
  
  * 設計架構圖  
  ![架構圖](https://github.com/Sapphire1002/VHDL/blob/main/08%20fpga_connection/20210311_%E8%A8%AD%E8%A8%88%E6%9E%B6%E6%A7%8B%E5%9C%96.PNG)  
  說明:  
  freq_div: 除頻  
  freq_clk: 除 2^23 訊號  
  in_out_data: 控制當前要輸出或接收資料  
  data:  為 inout 輸出輸入  
  count: 計算當前球的位置  
  serve: 控制發球  
  ena: 控制當前讀寫  
  
  * 當前實作結果  
  [影片連結](https://drive.google.com/file/d/164o2yVWDuCR0Ng5jTcPbP7ncfTkR8vN6/view?usp=sharing)  
  說明:  
  左邊的板子發球過去可以到對面  
  
  </details>
  
  <details>
  <summary> 問題討論 </summary>
  
   * inout 時序  
   * 傳送和接收訊號時會延遲 1 個 clk  
  - [ ] 已解決      
  - [x] 未解決  
  
  </details>

</details>

## 心臟影像
#### **每周進度**
+ **2021.05.07 ~ 2021.07.06 皆為練習學長教的內容**

  <details>
    <summary> 2021.05.07 ~ 2021.05.12 </summary>  
    進度:  

    1. 分割心臟肌肉位置  
    2. 了解心臟 10 個種類  
  程式碼: [split_muscle.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/split_muscle.py)
  </details>

  <details>
    <summary> 2021.05.13 ~ 2021.05.19 </summary>  
    進度:  

    1. 分割心臟肌肉位置
    2. 手動分類心臟種類
    3. 了解 DBSCAN  
    程式碼: [split_muscle_v2.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/split_muscle_v2.py)
  </details>

  <details>
    <summary> 2021.05.20 ~ 2021.05.26 </summary>  
    進度:  

    1. 將未分好類別的心臟圖片做分類
    2. 訓練分類器(跑過學長的程式)  
  程式碼:  
  [classification_datasets.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/classification_datasets.py)  
  [train.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/example/train.py)    
  [confusionMatrix.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/example/confusionMatrix.py)   
  </details>

  <details>
    <summary> 2021.05.27~ 2021.06.02 </summary>  
    進度:  

    1. 將 dcm 檔案轉 avi, png檔案
    2. 自行設計分類器  
      (嘗試 sklearn 裡面的其他方式，比較準確率及執行時間)  
    程式碼:  
    [dcm_avi.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/dcm_avi.py)  
    [check.txt](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/dcm_avi_data/check.txt)  
    [report.log](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/dcm_avi_data/report.log)  
  </details>

  <details>
    <summary> 2021.06.03 ~ 2021.06.09 </summary>  
    進度:  

    1. 自行設計分類器  
    程式碼: [classifier_test01.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_test01.py)  

  </details>

  <details>
    <summary> 2021.06.10 ~ 2021.06.16 </summary>  
    進度:  

    1. 自行設計分類器
    2. 檢查資料集  
    程式碼:  
    [classifier_test02.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_test02.py)  
    [test_model.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_use_test_model.py)  
    [confusionMat.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_test_model_confusionMat.py)  
    [classifier_tts_datasets.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_tts_datasets.py)  
    [check_datasets.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/check_datasets.py)  
  </details>

  <details>
    <summary> 2021.06.17 ~ 2021.06.23 </summary>  
    進度:  

    1. 研究 json 資料格式(含補足 TP, TN, FP, FN 資料)
    2. 自行設計分類器(使用不同分類器嘗試)
    3. 處理上周的問題紀錄和開會紀錄   
    程式碼:  
    [confusionMat.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_test_model_confusionMat.py)  
    [classifier_tts_datasets.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_tts_datasets.py)  
    [sklearn_model.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_sklearn.py)  
    [sklearn_model_confusionMat.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/sklearn_model_confusionMat.py)  
    [sklearn_rf.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/classifier_sklearn_rf.py)  
    [use_rf_model.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/my%20classifier/use_random%20forest_model.py)  

  </details>

  <details>
    <summary> 2021.06.24 ~ 2021.06.30 </summary>  
    進度:  

    1. 優化程式及計算程式執行時間  
    程式碼:  
    [dcm_avi.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/dcm_avi.py)  
    [train.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/example/train.py)  
  </details>

  <details>
    <summary> 2021.07.01 ~ 2021.07.06 </summary>  
    進度:  

    1. 優化骨架化程式及計算該程式執行時間  
    程式碼: [Models_optimize_v1.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/optimization/Models_optimize_v1.py)
  </details>

+ **2021.07.07 ~ 2021.07.27 Color Doppler**
  <details>
    <summary> 2021.07.07 ~ 2021.07.13 </summary>  
    進度:  

    1. 研究都卜勒效應
    2. 分割出影片中扇形區域
    3. 分出扇形區域中的顏色部分
    4. 新資料轉檔和骨架化  
    程式碼:  
  [doppler.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/doppler.py)  
  [check_avi_rename.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/check_avi_rename.py)  
  </details>

  <details>
    <summary> 2021.07.14 ~ 2021.07.20 </summary>  
    進度:  

    1. 分類檔案
    2. 分出扇形區域中的顏色部分
    3. 針對都卜勒彩色部分做處理  
    程式碼: [doppler2.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/doppler2.py)
  </details>

  <details>
    <summary> 2021.07.21 ~ 2021.07.27 </summary>  
    進度:  

    1. 分類檔案
    2. 針對都卜勒彩色部分做處理
    3. 將影像中白色扇形區域移除，還原原始影像  
    程式碼:  
    [doppler_curr2.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/doppler_curr2.py)  
    [restore_avi.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/restore_avi.py)

  </details>

+ **2021.07.28 ~ 2021.09.07 Parasternal Long Axis Matching & Watershed**
  <details>
    <summary> 2021.07.28 ~ 2021.08.03 </summary>  
    進度:  

    1. Match Parasternal Long Axis 模型  
    程式碼: [match_prac.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_prac.py)  
  </details>
  
  <details>
    <summary> 2021.08.04 ~ 2021.08.10 </summary>  
    進度:  

    1. 修正 Matching 模型的算法
    2. Matching 模型後做 Watershed 找出肌肉部分  
    程式碼: [match_prac.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_prac.py)    
  </details>
  
  <details>
    <summary> 2021.08.11 ~ 2021.08.17 </summary>  
    進度:  

    1. Matching 模型後做 Watershed 找出腔室部分
    2. ROI 區域調整  
    程式碼:  
    [match_prac.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_prac.py)  
    [find_roi.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/find_roi.py)
  </details>
  
  <details>
    <summary> 2021.08.18 ~ 2021.08.24 </summary>  
    進度:  

    1. Matching 模型後做 Watershed 找出腔室部分  
    程式碼: [match_prac4.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_prac.py)  
  </details>
  
  <details>
    <summary> 2021.08.25 ~ 2021.08.31 </summary>  
    進度:  

    1. 標記位置相對應關係
    2. 修正瓣膜抓取的方式  
    程式碼: [match_prac4.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_prac4.py)
  </details>
  
  <details>
    <summary> 2021.09.01 ~ 2021.09.07 </summary>  
    進度:  

    1. 處理都卜勒影像影響 watershed 的情況
    2. 修正同個位置 contour 不連續的算法  
    程式碼: [match_all](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/match_model/match_all(watershed))
  </details>

+ **2021.09.08 ~ 2021.09.20 Color Doppler 視覺化 & 演算法調整(月底醫生開會展示)**
  <details>
    <summary> 2021.09.08 ~ 2021.09.14 </summary>  
    進度:  

    1. Doppler 演算法調整、視覺化  
    程式碼:  
    [doppler_model](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/video/doppler_model/integration_v1.py)  
    [doppler_main.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/video/doppler_model/integration_main.py)
  </details>
  
  <details>
    <summary> 2021.09.15 ~ 2021.09.20 </summary>  
    進度:  

    1. Doppler 演算法調整、視覺化  
    程式碼:  
    [doppler_model](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/video/doppler_model/integration_v1.py)  
    [doppler_main.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Doppler/video/doppler_model/integration_main.py)
  </details>

+ **2021.09.21 ~ 2021.11.01 Parasternal Long Axis Segmentation & M-Mode(非確切醫學實際結果)**
  <details>
    <summary> 2021.09.21 ~ 2021.09.27 </summary>  
    進度:  

    1. 找標準長度
    2. Parasternal Long Axis M-Mode  
    程式碼:  
    [find_unit.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Unit/find_unit.py)  
    [PLA_M_Mode.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/m-mode/PLA_M_mode.py)
  </details>
  
  <details>
    <summary> 2021.09.28 ~ 2021.10.05 </summary>  
    進度:  

    1. PLA Segmentation  
    程式碼: [(matching_test)PLA_seg.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_all(muscle)/matching_test.py)  
  </details>
  
  <details>
    <summary> 2021.10.06 ~ 2021.10.12 </summary>  
    進度:  

    1. PLA Segmentation(M-Mode): 找出 Key Points  
    程式碼: [(M_Mode_matching)PLA_seg.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/m-mode/PLA_M_Mode_matching.py)  
  </details>
  
  <details>
    <summary> 2021.10.13 ~ 2021.10.19 </summary>  
    進度:  

    1. PLA Segmentation(M-Mode): 修正演算法  
    2. 研究 Speckle Tracking & GLS  
    程式碼: [(M_Mode_matching)PLA_seg.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/m-mode/PLA_M_Mode_matching.py)  
  </details>
  
  <details>
    <summary> 2021.10.20 ~ 2021.10.26 </summary>  
    進度:  

    1. Segmentation: 抓出心臟範圍
    2. 研究 Speckle Tracking & GLS
    3. 研究 CNN 兩篇論文的程式
    4. 研究 Condensation 演算法  
    程式碼: [(matching_test)heart_bound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_all(muscle)/matching_test.py)  
  </details>
  
  <details>
    <summary> 2021.10.27 ~ 2021.11.01 </summary>  
    進度:  

    1. Segmentation: 找出腔室中心點 & 邊界
    2. 論文程式架構圖  
    程式碼: [(matching_test)chamber_center.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/match_model/match_all(muscle)/matching_test.py)  
  </details>  

+ **2021.11.02 ~ 2021.11.08 整合模組**
  <details>
    <summary> 2021.10.27 ~ 2021.11.01 </summary>  
    進度:  

    1. 整合影像基本資訊模組  
       和檔案有關: DCM 轉 AVI (完成)  
       和影片有關: ROI、標準長度 & BPM(完成)  
    程式碼:  
  [find_unit4.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Unit/find_unit4.py)  
  [DCMToAVI.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Modular/DCMToAVI.py)
  </details>  

+ **2021.11.09 ~ 2021.12.21 Apical four chamber Segmentation & Mitral Valve**
  <details>
    <summary> 2021.11.09 ~ 2021.11.16 </summary>  
    進度:  

    1. 處理心臟週期    
    程式碼: [handle_period.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/handle_period.py)  
  </details>  
  
  <details>
    <summary> 2021.11.17 ~ 2021.11.23 </summary>  
    進度:  

    1. 處理心臟週期    
    2. Multi Threshold 一階插值  
    程式碼:   
  [handle_period.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/handle_period.py)  
  [multi_thres.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/multi_thres.py)
  </details> 
  
  <details>
    <summary> 2021.11.24 ~ 2021.11.30 </summary>  
    進度:  

    1. Multi threshold 完成公式部分  
    2. 抓取 A4C 心臟瓣膜位置  
    程式碼:   
  [multi_thres.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/multi_thres.py)  
  [find_valve.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/Find_valve.py)
  </details> 
  
  <details>
    <summary> 2021.12.01 ~ 2021.12.07 </summary>  
    進度:  

    1. Multi threshold 優化演算法  
    2. 修正 Kmeans 分群問題 & 初步預測腔室位置    
    程式碼:   
  [multi_thres.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/multi_thres.py)  
  [find_valve.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/Find_valve.py)
  </details> 
  
  <details>
    <summary> 2021.12.08 ~ 2021.12.14 </summary>  
    進度:  

    1. 用 C 語言 計算 Multi threshold 結合 Python  
    2. 調整 Multi threshold 計算 histogram 的方式  
    3. 修正 Kmeans 分群問題  
    4. 調整預測腔室位置的算法  
    5. 修正抓取瓣膜的演算法    
    程式碼:   
  [Multi_thres.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/Multi_Threshold/Multi_Threshold.py)  
  [convolution.c](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/Handle%20Convolution/convolution.c)  
  [FindValve_v2.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Segmentation/FindValve_v2.py)
  </details> 
  
  <details>
    <summary> 2021.12.15 ~ 2021.12.21 </summary>  
    進度:  

    1. 整合系統程式架構  
    程式碼: [Main.py](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System)  
  </details> 

+ **2021.12.22 ~ 2022.04.29 Apical four chamber View Global Longitudinal Strain**
1. #### Muscle Sampling 2021.12.22 ~ 2022.03.12
    <details>
      <summary> 2021.12.22 ~ 2021.12.28 </summary>  
      進度:  

      1. 利用骨架圖找心臟範圍  
      2. 整理問題紀錄  
      程式碼: [Segmentation_v2.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/System/Segmentation_v2.py)  
    </details> 

    <details>
      <summary> 2021.12.29 ~ 2022.01.04 </summary>  
      進度:  

      1. 研究 Anatomically based geometric modelling of the musculoskeletal system and other organs 論文   
    </details> 


    <details>
      <summary> 2022.01.05 ~ 2022.01.11 </summary>  
      進度:  

      1. 使用 Multi-Threshold 找心臟範圍  
      程式碼: [Heart_Bound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ReplaceSkeleton_test.py)    
    </details> 

    <details>
      <summary> 2022.01.12 ~ 2022.01.17 </summary>  
      進度:  

      1. 使用 Multi-Threshold 找心臟範圍  
      程式碼: [Heart_Bound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ReplaceSkeleton_test.py)    
    </details> 

    <details>
      <summary> 2022.01.18 ~ 2022.01.25 </summary>  
      進度:  

      1. 找出腔室間肌肉區域  
      2. 1st Multi-Threshold Result 白色區塊相連  
      3. 調整 Multi-Threshold  U、alpha、beta 值   
      程式碼:  
    [Heart_Bound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ReplaceSkeleton_test.py)   
      [ConnectHeartBound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound.py)
    </details> 

    <details>
      <summary> 2022.02.08 ~ 2022.02.15 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連    
         a. 定義可連接的區塊  
         b. 以 BFS 搜索可連接的節點  
      程式碼:  
    [ConnectBoundTest.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound_test.py)   
      [ConnectHeartBound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound.py)
    </details> 

    <details>
      <summary> 2022.02.16 ~ 2022.02.25 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連  
         a. 定義可連接的區塊  
         b. 將輪廓以特徵點取代  
         c. 以 BFS 搜索可連接的點  
      程式碼:  
    [ConnectBoundTest.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound_test.py)   
      [ConnectHeartBound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound.py)
    </details> 

    <details>
      <summary> 2022.02.26 ~ 2022.03.04 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連        
          a. 定義可連接的區塊(調整 Multi-Threshold 階數)    
          b. 將輪廓以特徵點取代(處理輪廓特徵點連接方式)   
          c. 以 BFS 搜索可連接的節點  
      程式碼:  
    [ConnectBoundTest.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound_test.py)   
      [ConnectHeartBound.py](https://github.com/Sapphire0912/MyProgramming/blob/master/Python/Project/implement/heart%20recognize/Heart%20Bound/ConnectHeartBound.py)
    </details> 

    <details>
      <summary> 2022.03.05 ~ 2022.03.11 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連        
          a. 定義可連接的區塊(調整 Multi-Threshold 階數)    
          b. 將輪廓以特徵點取代(處理輪廓特徵點連接方式)   
          c. 特徵點 Semantic & 預測 (基於已知 View)  
          - [x] 利用標準型的相對位置關係定義特徵點的位置    
          - [ ] 基於 Kmeans 的相對位置定義特徵點的位置  
      程式碼: [Semantic](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/Heart%20Bound/Semantic)   
    </details> 

2. #### LV Muscle Segmentation & Semantic(Speckle Tracking) 2022.03.12 ~ 2022.04.15
    <details>
      <summary> 2022.03.12 ~ 2022.03.17 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連        
          a. 定義可連接的區塊(調整 Multi-Threshold 階數)    
          b. 將輪廓以特徵點取代(處理輪廓特徵點連接方式)
          c. 疊加取平均後給機器學習模型訓練  
          d. 特徵點 Semantic & 預測 (基於已知 View)  
          - [ ] 利用標準型的相對位置關係定義特徵點的位置    
          - [x] 基於 Kmeans 的相對位置定義特徵點的位置    
      程式碼:  
    [Semantic](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/Heart%20Bound/Semantic)   
    [SVM_Train](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System2/Predict%20View)
    </details> 

    <details>
      <summary> 2022.03.18 ~ 2022.03.24 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連        
          a. 定義可連接的區塊(調整 Multi-Threshold 階數)    
          b. 將輪廓以特徵點取代(處理輪廓特徵點連接方式)
          c. 疊加取平均後給機器學習模型訓練  
          d. 特徵點 Semantic & 預測 (基於已知 View)  
          - [x] Matching(LV Region)   
          - [x] 基於 Kmeans 的瓣膜位置, 限制 LV 範圍  
      程式碼: [Matching](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/Matching)   
    </details> 

    <details>
      <summary> 2022.03.25 ~ 2022.03.31 </summary>  
      進度:  

      1. 1st Multi-Threshold Result 白色區塊相連        
          a. 定義可連接的區塊(調整 Multi-Threshold 階數)    
          b. 將輪廓以特徵點取代(處理輪廓特徵點連接方式)
          c. 疊加取平均後給機器學習模型訓練  
          d. 特徵點 Semantic & 預測 (基於已知 View)  
          - [x] Matching(LV Region)  
          - [x] 特徵點(Sampling) & 擬合曲線  
          - [x] 基於 Kmeans 的瓣膜位置, 限制 LV 範圍  
          - [x] 連接三段肌肉, 將點重新取樣   
      程式碼: [System3](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System3)   
    </details> 

    <details>
      <summary> 2022.04.02 ~ 2022.04.07 </summary>  
      進度:  

      1. 處理 Matching 的問題  
      2. GLS 定位點(Global Longitudinal Strain)  
      程式碼: [(Matching_v2)System3](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System3)   
    </details> 

    <details>
      <summary> 2022.04.08 ~ 2022.04.15 </summary>  
      進度:  

      1. 處理 Matching 的問題  
      2. 整理架構, 測試  
      程式碼: [(Matching_v2)System3](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System3)   
    </details> 
  
- #### 處理專題展內容 2022.04.16 ~ 2022.04.29
  <details>
    <summary> 2022.04.16 ~ 2022.04.22(因病調整進度) </summary>  
    進度:  

    1. 整理架構, 測試  
    2. PLA M-Mode  
    程式碼: [System2](https://github.com/Sapphire0912/MyProgramming/tree/master/Python/Project/implement/heart%20recognize/System2)   
  </details>

  <details>
    <summary> 2022.04.23 ~ 2022.04.29(處理專題展資料) </summary>  
    進度:  

    1. 處理專題展
    2. 收集 Global Longitudinal Strain 計算方法的資料  
  </details> 
