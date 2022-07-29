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
    char fname[200];

    // cap.get(CAP)

    size_t fn = 0;

    while (1)
    {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        sprintf(fname, "/Users/jose/Desktop/videos/test01/frame_%ld.jpg", fn);
        imwrite(fname, frame);
        fn += 1;

        imshow("frame", frame);

        char key = (char)waitKey(1);
        if (key == 27)
            break;
    }
    return 0;
}