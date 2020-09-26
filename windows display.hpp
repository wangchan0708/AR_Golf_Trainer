#include "stdafx.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <windows.h>
#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <ctype.h>
#include <iomanip>

#include <stdlib.h>

using namespace cv;
using namespace std;

void createwin()
{
    cvNamedWindow("Histogram", 0);
    cvNamedWindow("CamShift Demo", 1);
    setMouseCallback("CamShift Demo", onMouse, 0);
    createTrackbar("Vmin", "CamShift Demo", &vmin, 256, 0);
    createTrackbar("Vmax", "CamShift Demo", &vmax, 256, 0); //0代表函數不會在游標滑動時響應
    createTrackbar("Smin", "CamShift Demo", &smin, 256, 0);

    resizeWindow("CamShift Demo", 640, 480); //制定視窗大小
}

void imshowMany(const std::string &_winName, const vector<Mat> &_imgs, float x1)
{
    //destroyWindow(_winName);
    int nImg = (int)_imgs.size();

    Mat dispImg, dispImgResize;

    int x = 0, y = 0;

    float scale = 0.5;

    dispImg.create(Size(1310, 1000), CV_8UC3);

    //----------------------------------------------------------------------------
    //text display
    putText(dispImg, "club tracking", Point(30, 50), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));

    putText(dispImg, "club motion", Point(740, 50), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    rectangle(dispImg, Point(720, 60), Point(1250, 400), Scalar(255, 100, 0), 3, CV_AA);
    putText(dispImg, "Expected point of collision", Point(740, 100), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    putText(dispImg, "X1", Point(740, 150), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "Y1", Point(740, 200), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "velocity", Point(740, 270), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    putText(dispImg, "v1_x", Point(740, 320), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "v1_y", Point(740, 370), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));

    char text[255];
    /*for(int i=0;i<5;i++)
{
sprintf(text, "%d", i);
putText (dispImg, text, Point(800, 150),CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
}
*/
    sprintf(text, "%f", x1);
    putText(dispImg, text, Point(800, 150), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));

    putText(dispImg, "motion after collision", Point(740, 500), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    rectangle(dispImg, Point(720, 520), Point(1250, 845), Scalar(255, 100, 0), 3, CV_AA);
    putText(dispImg, "center of mass of ball ", Point(740, 550), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    putText(dispImg, "X2", Point(740, 600), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "Y2", Point(740, 650), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "velocity of ball ", Point(740, 720), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    putText(dispImg, "v2_x", Point(740, 770), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));
    putText(dispImg, "v2_y", Point(740, 820), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(255, 0, 215));

    putText(dispImg, "collision simulation", Point(30, 600), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0));
    //---------------------------------------------------------------------------

    for (int i = 0; i < nImg; i++)
    {
        x = _imgs[i].cols;
        y = _imgs[i].rows;

        if (i == 0)
        {
            Mat imgROI = dispImg(Rect(30, 70, x, y));
            resize(_imgs[i], imgROI, Size(x, y));
        }

        if (i == 1)
        {
            Mat imgROI = dispImg(Rect(_imgs[i - 1].cols + 60, 30, x, y));
            resize(_imgs[i], imgROI, Size(x, y));
        }

        if (i == 2)
        {
            Mat imgROI = dispImg(Rect(30, _imgs[i - 1].rows + 60, x, y));
            resize(_imgs[i], imgROI, Size(x, y));
        }
    }

    resize(dispImg, dispImgResize, Size(cvRound(dispImg.cols * scale), cvRound(dispImg.rows * scale)));

    namedWindow(_winName);
    imshow(_winName, dispImgResize);
}
