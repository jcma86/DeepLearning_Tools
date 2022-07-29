#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "../libs/helper/cmHelper.hpp"
#include "../libs/nn/cmNN.hpp"
#include "../libs/nn/cmActivationFunction.hpp"
#include "../libs/pso/cmPSO.hpp"

using namespace std;
using namespace cv;
using namespace cmNN;
using namespace cmPSO;

NeuralNetwork nn;
uint8_t *expected;
double lastBest = 99999999999999.99999;
double evaluateNN(size_t n, double *weights)
{
    nn.setWeights(weights);
    nn.compute(Z_SCORE);

    double *outNN = nn.getOutput();
    size_t outSize = nn.getOutputSize();

    size_t fitness = 0;
    for (size_t i = 0; i < outSize; i += 1)
        fitness += (uint8_t)(outNN[i] * 255.0) != expected[i] ? 1 : 0;

    return (double)fitness;
}

int main()
{
    char videoSrc[50];
    int inc = 5;
    size_t fi = 0;
    size_t fm = 5;
    size_t ff = 10;
    size_t nFrames = 0;
    size_t c, r;
    Mat tmp;

    double *nnInput;
    VideoCapture cap;

    sprintf(videoSrc, "/Users/jose/Desktop/video.mp4");

    cap.open(videoSrc, 0);
    nFrames = (size_t)cap.get(CAP_PROP_FRAME_COUNT);
    cap.set(CAP_PROP_POS_FRAMES, 1);
    cap >> tmp;
    c = tmp.cols * 0.3;
    r = tmp.rows * 0.3;
    size_t inputSize = c * r * 2;
    nnInput = new double[inputSize];

    printf("Video: %s\n", videoSrc);
    printf("Total frames: %ld\n", nFrames);
    printf("Frame size: %ldx%ld\n", c, r);

    namedWindow("First Frame");
    namedWindow("Middle Frame");
    namedWindow("Last Frame");
    moveWindow("First Frame", 50, 50);
    moveWindow("Middle Frame", 750, 200);
    moveWindow("Last Frame", 50, 500);

    /* NEURAL NETWORK CREATION - START */
    size_t neuronsPerLayer[5] = {1500, 300, 90, 90, c * r};
    char **fxs = (char **)malloc(4 * sizeof(char *));
    fxs[0] = (char *)malloc(15 * sizeof(char));
    fxs[1] = (char *)malloc(15 * sizeof(char));
    fxs[2] = (char *)malloc(15 * sizeof(char));
    fxs[3] = (char *)malloc(15 * sizeof(char));
    fxs[4] = (char *)malloc(15 * sizeof(char));
    strcpy(fxs[0], "fxLeakyReLU");
    strcpy(fxs[1], "fxLeakyReLU");
    strcpy(fxs[2], "fxLeakyReLU");
    strcpy(fxs[3], "fxLeakyReLU");
    strcpy(fxs[4], "fxIdentity");

    nn.createNeuronNetwork(inputSize, 5, neuronsPerLayer);
    nn.setInputs(nnInput);
    nn.setActivationFunction(&fxs);

    size_t countWeights = nn.getWeightsNeeded();
    /* NEURAL NETWORK CREATION - END */

    /* PSO CREATION - START */
    Swarm pso;
    size_t swarmDim = pso.createSwarm(5, countWeights, false);
    double *position = new double[swarmDim];
    double *velocity = new double[swarmDim];

    printf("Creating vectors %ld\n", swarmDim);
    cmHelper::Array::randomInit(swarmDim, position, -0.02, 0.02);
    cmHelper::Array::randomInit(swarmDim, velocity, -0.005, 0.005);
    printf("Creating vectors complete\n");

    pso.setFitnessFunction(evaluateNN);
    pso.initPosition(position, -1.0, 1.0);
    pso.initVelocity(velocity, -0.05, 0.05);
    pso.initWeights(0.729, 1.4944, 1.4944);
    /* PSO CREATION - END */

    fi = 0;
    fm = 5;
    ff = 10;
    Mat first;
    Mat middle;
    Mat last;

    Mat grayFirst;
    Mat grayMiddle;
    Mat grayLast;

    Mat doubleFirst;
    Mat doubleMiddle;
    Mat doubleLast;
    Mat generated;

    // for (int g = 0; g < 1000; g += 1)
    // {
    // while (ff < nFrames)
    // {
    cap.set(CAP_PROP_POS_FRAMES, fi);
    cap >> first;
    cap.set(CAP_PROP_POS_FRAMES, fm);
    cap >> middle;
    cap.set(CAP_PROP_POS_FRAMES, ff);
    cap >> last;

    resize(first, first, Size(c, r));
    resize(middle, middle, Size(c, r));
    resize(last, last, Size(c, r));

    cvtColor(first, grayFirst, COLOR_BGR2GRAY);
    cvtColor(middle, grayMiddle, COLOR_BGR2GRAY);
    cvtColor(last, grayLast, COLOR_BGR2GRAY);

    size_t totalPix = c * r;
    size_t in = 0;

    uint8_t *ptr = (uint8_t *)grayFirst.data;
    expected = (uint8_t *)grayMiddle.data;
    for (size_t i = 0; i < totalPix; i += 1, in += 1)
        nnInput[in] = (double)ptr[i] / 255.0;

    ptr = (uint8_t *)grayLast.data;
    for (size_t i = 0; i < totalPix; i += 1, in += 1)
        nnInput[in] = (double)ptr[i] / 255.0;

    for (int g = 0; g < 1000; g += 1)
    {
        pso.compute();
        printf("%4d - Best fitness: %d (%.2lf%%)\n", g, (int)pso.getBestFitness(), (100.0 * pso.getBestFitness()) / (r * c));
        if (lastBest != pso.getBestFitness())
        {
            lastBest = pso.getBestFitness();
            char path[150];
            sprintf(path, "bests/%d.txt", (int)lastBest);
            nn.saveToFIle(path);
        }
        pso.evolve();

        uint8_t *finalValues = new uint8_t[c * r];
        double *bp = pso.getBestPosition();
        for (size_t i = 0; i < (r * c); i += 1)
            finalValues[i] = (uint8_t)(bp[i] * 255.0);

        tmp = Mat(r, c, CV_8UC1, finalValues);

        imshow("First Frame", grayFirst);
        imshow("Middle Frame", tmp);
        imshow("Last Frame", grayLast);

        char key = (char)waitKey(1);
        if (key == 27)
            break;

        fi += inc;
        fm += inc;
        ff += inc;
        // }
    }

    destroyAllWindows();

    delete[] nnInput;
    delete[] position;
    delete[] velocity;

    return 0;
}