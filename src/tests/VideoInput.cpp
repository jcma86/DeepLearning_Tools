#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main()
{
    VideoCapture cap;

    cap.open("/Users/jose/Desktop/video.mov");

    while (1)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        imshow("frame", frame);

        char key = (char)waitKey(1);
        if (key == 27)
            break;
    }
    return 0;
}