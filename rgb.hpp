#include "stdafx.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//--------------------------------------------------------------------------------

using namespace cv;
using namespace std;

//--------------------------------------------------------------------------------

#include "stdafx.h"

//--------------------------------------------------------------------------------

int width = 640;
int height = 480;
Mat image(cv::Size(640, 480), CV_8UC3);
Mat hsv, hue, mask, histimg;
IplImage *img_out;
/*
CvHistogram *hist = 0;
CvHistogram *hist_0 = 0;
CvHistogram *hist_1 = 0;
CvHistogram *hist_2 = 0;
*/

Mat hist;

bool selectObject = false;
bool backprojMode = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
Rect trackWindow, trackWindow_temp;
RotatedRect trackBox; // tracking 返回的区域 box，带角度
CvConnectedComp track_comp;
int hdims = 80; // 划分HIST的个数，越高越精确
int hsize = 16;
float hranges[] = {0, 180};
const float *phranges = hranges;
int vmin = 10, vmax = 256, smin = 30;

//--------------------------------------
Point mousePosition; //这个用于储存 camshift 得到的 track_box.center.x and y

Point predict_pt; //这个就是 kalman 的预测坐标

const int winHeight = 640; //这个就是采集到的视频大小，这个写固定320 * 240  640 * 480
const int winWidth = 480;

bool bOnceSave = true; //保存数据只运行一次
int minWidth = 0;      //保存初始化时，跟踪的矩形框大小，之后跟踪的矩形框不能小于这个
int minHeight = 0;

//----------------------------------------
Point NowCursorPos; //存放当前的鼠标坐标
//Point OldCursorPos;
Point OldBox; //跟踪矩形框
Point NowBox;

int iOldSize = 0; //保存第一次运行的矩阵面积
int iNowSize = 0;

int iframe = 0; //统计帧数，每3帧数进行一次跟踪坐标的计算，获取一次当前鼠标位置，然后计算

//----------------------------------------

void onMouse(int event, int x, int y, int, void *) //定義滑鼠點擊
{
    if (selectObject)                   //當左鍵按下時開始圈選
    {                                   //矩形大小定位
        selection.x = MIN(x, origin.x); //讓滑鼠左右拉都可,只有origin的話只可右拉
        selection.y = MIN(y, origin.y);
        selection.width = abs(x - origin.x);
        selection.height = abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows); //確保矩形位在視窗中
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN: //左鍵按下
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        break;
    case EVENT_LBUTTONUP: //左鍵離開
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;
        break;
    }
}

CvScalar hsv2rgb(float hue)
{
    int rgb[3], p, sector;
    static const int sector_data[][3] =
        {{0, 2, 1}, {1, 2, 0}, {1, 0, 2}, {2, 0, 1}, {2, 1, 0}, {0, 1, 2}};
    hue *= 0.033333333333333333333333333333333f;
    sector = cvFloor(hue);
    p = cvRound(255 * (hue - sector));
    p ^= sector & 1 ? 255 : 0;

    rgb[sector_data[sector][0]] = 255;
    rgb[sector_data[sector][1]] = 0;
    rgb[sector_data[sector][2]] = p;

#ifdef _DEBUG
    printf("\n # Convert HSV to RGB：");
    printf("\n   HUE = %f", hue);
    printf("\n   R = %d, G = %d, B = %d", rgb[0], rgb[1], rgb[2]);
#endif

    return cvScalar(rgb[2], rgb[1], rgb[0], 0);
}

//读取Red初始化图片，以便進行tracking
bool loadTemplateImage_R()
{
    Mat tempimage = imread("d:/red.png");
    if (!tempimage.data)
    {
        return false;
    }

    cvtColor(tempimage, hsv, CV_BGR2HSV);
    int _vmin = vmin, _vmax = vmax;

    inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax), 0), Scalar(180, 256, MAX(_vmin, _vmax), 0), mask);
    Mat chan[3];
    split(hsv, chan);
    int ch[] = {0, 0};
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    selection.x = 1;
    selection.y = 1;
    selection.width = winHeight - 1; //640:480
    selection.height = winWidth - 1;

    Mat roi(hue, selection);      //得到ROI的選擇區域
    Mat maskroi(mask, selection); //mask保存的是hsv的最小值

    calcHist(&tempimage, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

    /*
	float max_val = 0.f;  
	minMaxLoc(hist, 0, &max_val, 0, 0);

	cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
	*/
    trackWindow = selection;
    trackObject = 1;

    return true;
}
//读取Green初始化图片，以便進行tracking
bool loadTemplateImage_G()
{
    Mat tempimage = imread("d:/green.png", 1);
    if (!tempimage.data)
    {
        return false;
    }

    cvtColor(tempimage, hsv, CV_BGR2HSV);
    int _vmin = vmin, _vmax = vmax;

    inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax), 0), Scalar(180, 256, MAX(_vmin, _vmax), 0), mask);
    Mat chan[3];
    //int chan[]={0,0,0};
    split(hsv, chan);
    int ch[] = {0, 0};
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    selection.x = 1;
    selection.y = 1;
    selection.width = winHeight - 1; //640:480
    selection.height = winWidth - 1;

    Mat roi(hue, selection);      //得到ROI的選擇區域
    Mat maskroi(mask, selection); //mask保存的是hsv的最小值

    calcHist(&tempimage, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

    /*
	float max_val = 0.f;  
	minMaxLoc(hist, 0, &max_val, 0, 0);

	cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
	*/

    trackWindow = selection;
    trackObject = 1;

    return true;
}
//读取Blue初始化图片，以便進行tracking
bool loadTemplateImage_B()
{
    Mat tempimage = imread("d:/blue.png", 1);
    if (!tempimage.data)
    {
        return false;
    }

    cvtColor(tempimage, hsv, CV_BGR2HSV);
    int _vmin = vmin, _vmax = vmax;

    inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax), 0), Scalar(180, 256, MAX(_vmin, _vmax), 0), mask);
    Mat chan[3];

    //int chan[]={0,0,0};
    *(Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    split(hsv, chan);
    int ch[] = {0, 0};
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    selection.x = 1;
    selection.y = 1;
    selection.width = winHeight - 1; //640:480
    selection.height = winWidth - 1;

    Mat roi(hue, selection);      //得到ROI的選擇區域
    Mat maskroi(mask, selection); //mask保存的是hsv的最小值

    calcHist(&tempimage, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, CV_MINMAX);

    /*
	float max_val = 0.f;  
	minMaxLoc(hist, 0, &max_val, 0, 0);

	cvConvertScale( hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0 );
	*/

    trackWindow = selection;
    trackObject = 1;

    return true;
}

//减法求绝对值的
int iAbsolute(int a, int b)
{
    int c = 0;
    if (a > b)
    {
        c = a - b;
    }
    else
    {
        c = b - a;
    }
    return c;
}
