#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main()
{
    Mat frame, converted;
    VideoCapture cap;

    int deviceID = 0;        // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API

    cap.open(deviceID, apiID);

    if (!cap.isOpened())
    {
        return -1;
    }

    for (;;)
    {
        cap.read(frame);
        if (frame.empty())
        {
            break;
        }

        cvtColor(frame, converted, COLOR_RGBA2GRAY);
        // frame.convertTo(converted, CV_COLO);

        imshow("Live", frame);
        imshow("Live2", converted);

        if (waitKey(5) >= 0)
            break;
    }

    return 0;
}
