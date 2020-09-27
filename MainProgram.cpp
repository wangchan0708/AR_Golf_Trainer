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

    int count = 0;
    int DELAY_CAPTION = 1500;
    int DELAY_BLUR = 100;
    int MAX_KERNEL_LENGTH = 31;

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
    
    //先初始化kalman (kalman filter setup)
    //Mat img(500, 500, CV_8UC3);
	KalmanFilter KF(4, 2, 0);
    Mat state(4, 1, CV_32F); // state(x,y,deltaX,deltaY)
    Mat processNoise(4, 1, CV_32F);
    Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));//measurement(x,y)
	KF.statePre.at<float>(0) = mousePosition.x;
    KF.statePre.at<float>(1) = mousePosition.y;
                    

    char code = (char)-1;
	randn( state, Scalar::all(0), Scalar::all(0.1) );
	
	/*
                | 1 0 1 0 |                | 0.2  0   0.2   0  |
     TransMat = | 0 1 0 1 |     Process =  |  0  0.2   0   0.2 |
                | 0 0 1 0 |                |  0   0   0.3   0  |
                | 0 0 0 1 |                |  0   0    0   0.3 |
    */
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1); // Including velocity
	KF.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,  0,0,0,0.3);

	//memcpy(KF->transitionMatrix->data.fl,A,sizeof(A)); 
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));
	//initialize post state of kalman filter at random
    randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));


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

                
                //速度值運算
                float x = trackBox.center.x, y = trackBox.center.y;
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


                char filename[] = "homography.xml"; //floor - cam - projector 的轉換矩陣
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

                mousePosition.x = trackBox.center.x;
		        mousePosition.y = trackBox.center.y;
			    
                
                //進行进行 kalman 预测，可以得到 predict_pt 预测坐标
			    //2.kalman prediction
                Mat prediction = KF.predict();
                randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
			    Point predict_pt(prediction.at<float>(0),prediction.at<float>(1));
			
                //3.update measurement
			    measurement(0) = mousePosition.x;
                measurement(1) = mousePosition.y;

			    Point measPt(measurement(0),measurement(1));

			    //4.update
			    Mat estimated = KF.correct(measurement);
                Point statePt(estimated.at<float>(0),estimated.at<float>(1));
			    //因为视频设置的采集大小是640 * 480，那么这个搜索区域夸大后也得在这范围内

                //==========================================================================
	    		//下面这里其实就是一个粗略的预测 trackWindow 的范围
			    //因为在鼠标选取的时候，有时可能只是点击了窗体，所以没得 width  和 height 都为0
			    //==========================================================================
                int iBetween = 0;
			    //确保预测点 与 实际点之间 连线距离 在 本次 trackBox 的size 之内
			    iBetween = sqrt(powf((predict_pt.x - trackBox.center.x),2) + powf((predict_pt.y- trackBox.center.y),2) );

			    CvPoint prePoint;//预测的点 相对于 实际点 的对称点

			    if ( iBetween > 5)
			    {
				    //当实际点 在 预测点 右边
				    if (trackBox.center.x > predict_pt.x)
				    {
					    //且，实际点在 预测点 下面
					    if (trackBox.center.y > predict_pt.y)
					    {
						    prePoint.x = trackBox.center.x + iAbsolute(trackBox.center.x,predict_pt.x);
						    prePoint.y = trackBox.center.y + iAbsolute(trackBox.center.y,predict_pt.y);
					    }
					    //且，实际点在 预测点 上面
					    else
					    {
						    prePoint.x = trackBox.center.x + iAbsolute(trackBox.center.x,predict_pt.x);
						    prePoint.y = trackBox.center.y - iAbsolute(trackBox.center.y,predict_pt.y);
					    }
					    //宽高
					    if (trackWindow.width != 0)
					    {
						    trackWindow.width += iBetween + iAbsolute(trackBox.center.x,predict_pt.x);
					    }

					    if (trackWindow.height != 0)
					    {
						    trackWindow.height += iBetween + iAbsolute(trackBox.center.x,predict_pt.x);
					    }
				    }
				    //当实际点 在 预测点 左边
				    else
				    {
					    //且，实际点在 预测点 下面
					    if (trackBox.center.y > predict_pt.y)
					    {
						    prePoint.x = trackBox.center.x - iAbsolute(trackBox.center.x,predict_pt.x);
						    prePoint.y = trackBox.center.y + iAbsolute(trackBox.center.y,predict_pt.y);
					    }
					    //且，实际点在 预测点 上面
					    else
					    {
						    prePoint.x = trackBox.center.x - iAbsolute(trackBox.center.x,predict_pt.x);
						    prePoint.y = trackBox.center.y - iAbsolute(trackBox.center.y,predict_pt.y);
					    }
					    //宽高
					    if (trackWindow.width != 0)
					    {
						    trackWindow.width += iBetween + iAbsolute(trackBox.center.x,predict_pt.x);
					    }

					    if (trackWindow.height != 0)
					    {
						    trackWindow.height += iBetween +iAbsolute(trackBox.center.x,predict_pt.x);
					    }
				    }

				    trackWindow.x = prePoint.x - iBetween;	
				    trackWindow.y = prePoint.y - iBetween;
			    }
			    else
			    {
				    trackWindow.x -= iBetween;
				    trackWindow.y -= iBetween;
				    //宽高
				    if (trackWindow.width != 0)
				    {
					    trackWindow.width += iBetween;
				    }

				    if (trackWindow.height != 0)
				    {
					    trackWindow.height += iBetween;
				    }
			    }

			    //跟踪的矩形框不能小于初始化检测到的大小，当这个情况的时候，X 和 Y可以适当的在缩小
			    minWidth = trackBox.size.width;
		        minHeight = trackBox.size.height;
			    if (trackWindow.width < minWidth)
			    {
				    trackWindow.width = minWidth;
				    trackWindow.x -= iBetween;
			    }
			    if (trackWindow.height < minHeight)
			    {
		    		trackWindow.height = minHeight;
			    	trackWindow.y -= iBetween;
			    }

			    //确保调整后的矩形大小在640 * 480之内
	    		if (trackWindow.x <= 0)
		    	{
		    		trackWindow.x = 0;
		    	}
			    if (trackWindow.y <= 0)
			    {
			    	trackWindow.y = 0;
			    }
		    	if (trackWindow.x >= 600)
		    	{
		    		trackWindow.x = 600;
		    	}
		    	if (trackWindow.y >= 440)
		    	{
			    	trackWindow.y = 440;
			    }

			    if (trackWindow.width + trackWindow.x >= 640)
			    {
				    trackWindow.width = 640 - trackWindow.x;
		    	}
		    	if (trackWindow.height + trackWindow.y >= 640)
		    	{
                  trackWindow.height = 640 - trackWindow.y;
		    	}
			
			//------------------------------------------------------------------------------------------
			    img_out=cvCreateImage(cvSize(winWidth,winHeight),8,3);
			    cvSet(img_out,cvScalar(255,255,255,0));
			    char buf[256];
			    sprintf(buf,"%d",iBetween);
			    cvPutText(img_out,buf,cvPoint(10,30),&font,CV_RGB(0,0,0));
			    sprintf(buf,"%d : %d",trackWindow.x,trackWindow.y);
			    cvPutText(img_out,buf,cvPoint(10,50),&font,CV_RGB(0,0,0));
			    sprintf(buf,"%d : %d",trackWindow.width,trackWindow.height);
			    cvPutText(img_out,buf,cvPoint(10,70),&font,CV_RGB(0,0,0));

			    sprintf(buf,"size: %0.2f",trackBox.size.width * trackBox.size.height);
			    cvPutText(img_out,buf,cvPoint(10,90),&font,CV_RGB(0,0,0));
			//-------------------------------------------------------------------------------------
			
			/*if( image->origin )
				trackBox.angle = -trackBox.angle;
				*/
		    	POINT OldCursorPos;
		  
			    if (iframe == 0)
			    {   
                    GetCursorPos(&OldCursorPos);
				    OldBox.x = trackBox.center.x;
				    OldBox.y = trackBox.center.y;
		    	}
			    if (iframe < 3)//每3帧进行一次判断
			    {
				    iframe++;
    			}
	    		else
		    	{
			    	iframe = 0;
				    iNowSize = trackBox.size.width * trackBox.size.height;

	    			if ((iNowSize / iOldSize) > (3/5))
		    		{
			    		NowBox.x = trackBox.center.x;
				    	NowBox.y = trackBox.center.y;

					    SetCursorPos(OldCursorPos.x - (NowBox.x - OldBox.x)*1366/640,
					    OldCursorPos.y - (NowBox.y - OldBox.y)*768/480);
				    }
			    }
			//--------------------------------------------------------------------------
			
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
