#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "../libs/neural_networks/cmNeuralNetwork.hpp"
#include "../libs/neural_networks/cmActivationFunction.hpp"

using namespace cv;
using namespace cmNeuralNetwork;

int main()
{
    int nLayers = 25;
    Layer *layer = new Layer[nLayers];

    double *pixelsD;
    double *pixelsPtr;

    int c = 240;
    int r = 160;

    Mat img = imread("/Users/jose/me.jpg", 0);
    Mat imgDouble;
    Mat out;
    Mat finalImg;

    resize(img, img, Size(c, r));
    img.convertTo(imgDouble, CV_64FC1);
    pixelsPtr = (double *)imgDouble.data;

    pixelsD = new double[c * r];
    for (int i = 0; i < c * r; i += 1)
        pixelsD[i] = pixelsPtr[i] / 255.0;

    printf("Creating layers.\n");
    layer[0].createLayer(200);
    for (int i = 1; i < nLayers - 1; i += 1)
        layer[i].createLayer(100);
    layer[nLayers - 1].createLayer(c * r);

    printf("Setting inputs.\n");
    layer[0].setInputs(c * r, pixelsD);
    for (int i = 1; i < nLayers; i += 1)
        layer[i].setInputs(layer[i - 1].getOutputSize(), layer[i - 1].getOtput());

    // layer[nLayers - 1].setExtraInputs(c * r, pixelsD);

    printf("Setting activation functions.\n");
    for (int i = 0; i < nLayers - 1; i += 1)
        layer[i].setActivationFunction(fxTanh);
    layer[nLayers - 1].setActivationFunction(fxSigmoid);

    size_t nw = 0;
    for (int i = 0; i < nLayers; i += 1)
        nw += layer[i].weightsNeeded();

    printf("Generating weights (%ld).\n", nw);
    double *w = new double[nw];
    NNHelper::randomWeights(nw, w);

    size_t firstW = 0;
    for (int i = 0; i < nLayers; i += 1)
    {
        layer[i].setWeights(layer[i].weightsNeeded(), &w[firstW]);
        firstW += layer[i].weightsNeeded();
        printf("Setting weights (%ld).\r", firstW);
    }

    printf("\nComputing\n");
    for (int i = 0; i < nLayers; i += 1)
        layer[i].compute();

    double *outNN = layer[nLayers - 1].getOtput();
    for (int i = 0; i < (r * c); i += 1)
    {
        outNN[i] *= 255.0;
        printf("%lf\n", outNN[i]);
    }

    out = Mat(r, c, CV_64FC1, outNN);
    out.convertTo(finalImg, CV_8UC1);

    imshow("In", img);
    imshow("Out", finalImg);

    waitKey(0);

    delete[] pixelsD;
    delete[] w;
    delete[] layer;

    return 0;
}
