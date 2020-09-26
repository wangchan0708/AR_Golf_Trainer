#include "stdafx.h"
#include "rgb.hpp" //滑鼠動作設定 & RGB純色設定
#include "windows display.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <windows.h>
#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <ctype.h>
#include <iomanip>

#include <stdlib.h>  
#include <time.h> 

using namespace cv;
using namespace std;

int main( int argc, const char** argv ){


    Mat frame;

	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_SCRIPT_COMPLEX,1,1);

    //open cam or viedo
    VideoCapture cap(0); // open the video camera no. 0
    //VideoCapture cap("D:/VideoTest.avi");   //讀取視頻檔
    
	if( !cap.isOpened() )//exception handling if cannnot open the camera
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }

	//camshift 定義
	VideoWriter writer("VideoTest.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));  //將攝影內容轉成avi輸出
    Rect trackWindow;
    RotatedRect trackBox; //定義旋轉矩陣
    int hsize = 32;
    float hranges[] = {0,180};
    const float* phranges = hranges;
	
	createwin(); //建立視窗

    Mat  hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    float a=0,b=0;

	//infinite loop start
    for (;;)
    {  
        long i = 10000000L;
        clock_t start, finish;
        double duration;
        start = clock(); //第一次取時間

        if (!paused) //没有暂停
        {
            cap >> frame; //從攝影機抓取圖像至frame
            //writer<< frame;
            writer << image;
            if (frame.empty())
                break;
        }

        frame.copyTo(image);
        //-------------------Camshift主程式-------------------------------------------------------------------------------
        if (!paused)
        {
            cvtColor(image, hsv, CV_BGR2HSV); //RBG轉成HSV

            if (trackObject) //選取追蹤目標物:初始為0,滑鼠單擊後鬆開變-1
            {
                int _vmin = vmin, _vmax = vmax;
                
                /*檢查HSV的三通道是否都在範圍內
                是的話mask為1(src的值複製給dst),反之則為0(Note:不論src為何dst都是0)*/
                inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);

                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());  //將hue初始化成和hsv相同大小深度的矩陣
                mixChannels(&hsv, 1, &hue, 1, ch, 1); //將hsv的第一個數據也就是色相(hue)複製到hue

                //--------------------------------------------------------------------------
                //此if loop只會執行一次(用來定義圈選的顏色直方圖與矩形繪製)

                if (trackObject < 0) //滑鼠點擊後開始執行此if迴圈
                {
                    Mat roi(hue, selection);      //得到ROI的選擇區域
                    Mat maskroi(mask, selection); //mask保存的是hsv的最小值
                    //roi與指標數據共用相同儲存區,換句話說在roi的操作也會作用於hue上
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    //calchist用來計算陣列的直方圖
                    /*第一格參數為輸入的陣列
                      第二為輸入的矩陣數目
                      第三為計算直方圖的通道
                      第四:不為0的遮罩決定那些元素與直方圖計算
                      第五為輸出的直方圖hist
                      第六為直方圖的維數
                      第七為每一維度的直方陣列大小.第八為豎座標顯示範圍*/
                    normalize(hist, hist, 0, 255, CV_MINMAX);
                    //歸一化hist的範圍為0~255

                    trackWindow = selection;
                    trackObject = 1; //除非按'c'歸零否則此if只會執行一次

                    histimg = Scalar::all(0); //all(0)將所有數值歸零
                    int binW = histimg.cols / hsize;

                    Mat buf(1, hsize, CV_8UC3);
                    //CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels.
                    for (int i = 0; i < hsize; i++)
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / hsize), 255, 255);
                    //vec3b為一個3 char的向量,saturate_cast用來防止數據溢出(小於最小值變為最小值,大於最大值變為最大值)
                    //at是返回一個指定格式的參考值
                    cvtColor(buf, buf, CV_HSV2BGR); //HSV轉回為RGB

                    for (int i = 0; i < hsize; i++)
                    {
                        int val = saturate_cast<int>(hist.at<float>(i) * histimg.rows / 255);
                        //在輸入的圖像上畫矩形,指定左上和右下,並定義顏色大小及線形等
                        rectangle(histimg, Point(i * binW, histimg.rows),
                                  Point((i + 1) * binW, histimg.rows - val),
                                  Scalar(buf.at<Vec3b>(i)), -1, 8);
                    }
                }
                //------------------------------------------------------------------------------------------

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                //計算hue的0通道直方圖hist的反向投影,並存到backproj裡
                backproj &= mask;

                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                                TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1)); //TermCriteria是迭代終止的條件

                //---------------------------------------------
                //保存最小要求的矩形框大小
                /*
				if (bOnceSave){
				minWidth = trackBox.size.width;
				minHeight = trackBox.size.height;
				iOldSize = minHeight * minWidth;
				bOnceSave = false;
			    }
			    */
            

                if (backprojMode)
                    cvtColor(backproj, image, CV_GRAY2BGR);

                ellipse(image, trackBox, Scalar(0, 0, 255), 2, CV_AA); //用橢圓做為追蹤的框架

                
                if (trackWindow.area() <= 100) //設定area<=10000時停止追蹤
                {
                    trackObject = 0;
                    histimg = Scalar::all(0);
                }

                //----------------------------------------
                //速度值運算

                float x = trackBox.center.x, y = trackBox.center.y;
                // cout<<"(x:"<<x<<",y:"<<y<<")"<<endl;

                float daltCam_x, daltCam_y;
                float vx, vy;

                Point2f p1;
                p1.x = a, p1.y = b;

                daltCam_x = x - a;
                daltCam_y = y - b;

                
                //從camera轉至floor
                //Mat deltCam=(Mat_<float>(3,1)<<daltCam_x,daltCam_y,1); //轉算前矩陣定義
                //Mat deltFloor;//宣告轉換後的矩陣定義

                Point2f deltCam_p(daltCam_x, daltCam_y);
                Point2f deltFloor_p;
                Mat_<Point2f> deltCam(1, 1, deltCam_p); //vector to mat
                Mat deltFloor;                          //宣告轉換後的矩陣定義


                char filename[] = "homography.xml";
                Mat H21, H22;
                FileStorage fs(filename, CV_STORAGE_READ); //open file storage

                fs["H21"] >> H21; //use camera 3
                fs["H22"] >> H22;

                perspectiveTransform(deltCam, deltFloor, H22); //each element should be 2D/3D data (大矩陣裡包了很多小矩陣) {[],[],[]......}
                deltFloor_p = deltFloor.at<Point2f>(0);
        
                cout << deltFloor.at<Point2f>(0) << endl;

                vx = (deltFloor_p.x) / duration;
                vy = (deltFloor_p.y) / duration;

                cout << deltFloor_p.x << endl;

                a = x, b = y;

                arrowedLine(image, p1, trackBox.center, Scalar(255, 0, 0), 5, CV_AA); //將速度方向視覺化(箭頭)

            }
        }

        //========================================================================================
        /*後面的code不管一開始paused是true/false,都要執行*/
        else if (trackObject < 0)
            paused = false;

        if (selectObject && selection.width > 0 && selection.height > 0)
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi); //bitwise_not是將每個bit位取反的函數
        }

        imshow("CamShift Demo", image);
        imshow("Histogram", histimg);

        vector<Mat> imgs(1);
        imgs[0] = image;
        imshowMany("mainWindows", imgs, trackBox.center.x);


        //---------------------------------鍵盤控制列------------------------------------
        char c = (char)waitKey(10);
        if (c == 27)
            break;
        switch (c) //退出鍵
        {
        case 'b': //反向投影交替
            backprojMode = !backprojMode;
            break;
        case 'c': //清出追蹤對象
            trackObject = 0;
            histimg = Scalar::all(0);
            break;
        case 'h':
            showHist = !showHist;
            if (!showHist)
                destroyWindow("Histogram");
            else
                namedWindow("Histogram", 1);
            break;
        case 'p': //暫停追蹤
            paused = !paused;
            break;
        default:;
        }
        //-------------------------------------------------------------------------------------
        
        //第二次取時間,並計算與第一次的差距
        finish = clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC; //單位為秒
                                                              //printf( "%f seconds\n", duration );
    }
    //infinite loop finish

    return 0;
}
