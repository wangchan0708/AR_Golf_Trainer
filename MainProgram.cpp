#include "stdafx.h"
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
    //VideoCapture cap("D:/VideoTest.avi"); 
    
	if( !cap.isOpened() )//exception handling if cannnot open the camera
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }


return 0;
}