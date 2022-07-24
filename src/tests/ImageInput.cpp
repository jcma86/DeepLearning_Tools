#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main()
{
    double *pixelsD;
    uint8_t *pixelsU;

    Mat imgDouble;
    Mat img = imread("/Users/jose/me.jpg", 0);

    resize(img, img, Size(800, 600));
    img.convertTo(imgDouble, CV_64FC1);
    pixelsD = (double *)imgDouble.data;
    pixelsU = (uint8_t *)img.data;

    imshow("Input", img);

    // pixelPtr = (uint8_t*)img.data; <-- Pointer to data
    // cn = img.channels();
    // i -> row .... j -> col
    // bgrPixel.val[0] = pixelPtr[i*foo.cols*cn + j*cn + 0]; // B
    // bgrPixel.val[1] = pixelPtr[i*foo.cols*cn + j*cn + 1]; // G
    // bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R
    for (int i = 0; i < (img.rows * img.cols); i += 1)
    {
        printf("%lf vs %d\n", pixelsD[i], pixelsU[i]);
    }

    waitKey(0);

    return 0;
}