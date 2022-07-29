#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "../libs/nn/cmNN.hpp"
#include "../libs/nn/cmActivationFunction.hpp"
#include "../libs/helper/cmHelper.hpp"

using namespace cv;
using namespace cmNN;

int main()
{
    // int nLayers = 50;
    // Layer *layer = new Layer[nLayers];

    // double *pixelsD;
    // double *pixelsPtr;

    // int c = 240;
    // int r = 160;

    // Mat img = imread("/Users/jose/me.jpg", 0);
    // Mat imgDouble;
    // Mat out;
    // Mat finalImg;

    // resize(img, img, Size(c, r));
    // img.convertTo(imgDouble, CV_64FC1);
    // pixelsPtr = (double *)imgDouble.data;

    // pixelsD = new double[c * r];
    // for (int i = 0; i < c * r; i += 1)
    //     pixelsD[i] = pixelsPtr[i] / 255.0;

    // printf("Creating layers.\n");
    // layer[0].createLayer(0, 2000);
    // for (int i = 1; i < nLayers - 1; i += 1)
    //     layer[i].createLayer(i, 500);
    // layer[nLayers - 1].createLayer(nLayers - 1, c * r);

    // printf("Setting inputs.\n");
    // layer[0].setInputs(c * r, pixelsD);
    // for (int i = 1; i < nLayers; i += 1)
    //     layer[i].setInputs(layer[i - 1].getOutputSize(), layer[i - 1].getOtput());

    // // layer[nLayers - 1].setExtraInputs(c * r, pixelsD);

    // printf("Setting activation functions.\n");
    // for (int i = 0; i < nLayers - 1; i += 1)
    //     layer[i].setActivationFunction("fxLeakyReLU");
    // layer[nLayers - 1].setActivationFunction("fxIdentity");

    // size_t nw = 0;
    // for (int i = 0; i < nLayers; i += 1)
    //     nw += layer[i].weightsNeeded();

    // printf("Generating weights (%ld).\n", nw);
    // double *w = new double[nw];
    // cmHelper::Array::randomInit(nw, w);

    // size_t firstW = 0;
    // for (int i = 0; i < nLayers; i += 1)
    // {
    //     layer[i].setWeights(layer[i].weightsNeeded(), &w[firstW]);
    //     firstW += layer[i].weightsNeeded();
    //     printf("Setting weights (%ld).\r", firstW);
    // }

    // printf("\nComputing\n");
    // for (int i = 0; i < nLayers - 1; i += 1)
    //     layer[i].compute(Z_SCORE);
    // layer[nLayers - 1].compute(MIN_MAX, false);

    // double *outNN = layer[nLayers - 1].getOtput();
    // uint8_t *finalValues = new uint8_t[c * r];
    // for (int i = 0; i < (r * c); i += 1)
    // {
    //     finalValues[i] = (uint8_t)(outNN[i] * 255.0);
    //     printf("%.15lf vs %d\n", outNN[i] * 255.0, finalValues[i]);
    // }

    // out = Mat(r, c, CV_8UC1, finalValues);

    // imshow("In", img);
    // imshow("Out", out);

    // waitKey(0);

    // delete[] pixelsD;
    // delete[] w;
    // delete[] layer;
    // delete[] finalValues;

    return 0;
}
