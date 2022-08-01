#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>

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

typedef struct
{
    unsigned int x1;
    unsigned int y1;
    unsigned int width;
    unsigned int height;

    unsigned int centerX;
    unsigned int centerY;
    unsigned int x2;
    unsigned int y2;

    unsigned int blur;
    unsigned int expression;
    unsigned int illumination;
    unsigned int occlusion;
    unsigned int pose;
    unsigned int invalid;
} Face;

typedef struct
{
    string path;
    vector<Face> faces;
} ExampleInfo;

double lastBest = -99999999999999.99999;

vector<ExampleInfo> examples;
NeuralNetwork nn;
Swarm pso;
size_t validFaces = 0;
size_t invalidFaces = 0;
double *nnInput = NULL;

void buildExpectedResult(ExampleInfo example)
{
    Mat imgExample = imread("/Users/jose/Downloads/WIDER_DATASET/training/" + example.path, 0);

    for (size_t f = 0; f < example.faces.size(); f += 1)
    {
        // circle(imgExample, Point2d(example.faces[f].x1 + example.faces[f].width / 2, example.faces[f].row + example.faces[f].height / 2), 5, 250, 2);
    }

    // rectangle(imgExample, Point2d(example.faces[0].column, example.faces[0].row), Point2d(example.faces[0].column + example.faces[0].width, example.faces[0].row + example.faces[0].height), 0, 3);
    imshow("Example", imgExample);

    waitKey(0);
}

double trainNN(void *psoParams)
{
    psoFitnessFxParams *params = (psoFitnessFxParams *)psoParams;
    double fitness = 0.0;

    nn.setWeights(params->position);

    // char windowName[50];
    // sprintf(windowName, "Face Example For Particle %ld", params->particleID);

    // namedWindow(windowName);
    // moveWindow(windowName, 50, 50);

    size_t correctValid = 0;
    size_t incorrectValid = 0;

    size_t correctInvalid = 0;
    size_t incorrectInvalid = 0;

    size_t totalExamples = examples.size();
    for (size_t e = 0; e < totalExamples; e += 1)
    {
        Mat img = imread("/Users/jose/Downloads/WIDER_DATASET/training/" + examples[e].path, 0);
        uint8_t *ptr;

        size_t totalFaces = examples[e].faces.size();
        for (size_t f = 0; f < totalFaces; f += 1)
        {
            Face *face = &examples[e].faces[f];
            Rect crop(face->x1, face->y1, face->width, face->height);
            Mat faceImg = img(crop);

            resize(faceImg, faceImg, Size(30, 30));

            ptr = (uint8_t *)faceImg.data;

            for (size_t i = 0; i < nn.getInputSize(); i += 1)
                nnInput[i] = (double)ptr[i] / 255.0;

            // imshow(windowName, faceImg);

            nn.compute(Z_SCORE);
            double *outNN = nn.getOutput();
            size_t outSize = nn.getOutputSize();

            if (outNN[0] >= 0.9)
            {
                correctValid += face->invalid == 0 ? 1 : 0;
                incorrectInvalid += face->invalid == 1 ? 1 : 0;
            }
            else
            {
                correctInvalid += face->invalid == 1 ? 1 : 0;
                incorrectValid += face->invalid == 0 ? 1 : 0;
            }

            // waitKey(1);
        }
    }
    // destroyAllWindows();

    double a = (double)correctValid / (double)validFaces;
    double b = (double)correctInvalid / (double)invalidFaces;

    double c = (double)incorrectValid / (double)validFaces;
    double d = (double)incorrectInvalid / (double)invalidFaces;

    fitness = ((0.5 * a) + (0.5 * b) - (0.5 * c) - (0.5 * d)) * 100.0;

    return fitness;
}

int main()
{
    fstream file;
    file.open("/Users/jose/Downloads/WIDER_DATASET/wider_face_split/wider_face_train_bbx_gt.txt", ios::in);

    if (!file.is_open())
        return -1;

    string line;
    size_t count = 0;
    while (getline(file, line))
    {
        ExampleInfo anExample;
        anExample.path.assign(line);
        getline(file, line);
        int nFaces = atoi(line.c_str());
        if (nFaces == 0)
            getline(file, line);
        for (int e = 0; e < nFaces; e += 1)
        {
            Face aFace;
            getline(file, line);
            istringstream lineToParse(line);
            lineToParse >> aFace.x1 >> aFace.y1 >> aFace.width >> aFace.height;
            lineToParse >> aFace.blur >> aFace.expression >> aFace.illumination >> aFace.invalid >> aFace.occlusion >> aFace.pose;
            aFace.x2 = aFace.x1 + aFace.width;
            aFace.y2 = aFace.y1 + aFace.height;
            aFace.centerX = aFace.x1 + (aFace.width / 2);
            aFace.centerY = aFace.y1 + (aFace.height / 2);

            if (aFace.x1 == aFace.x2 || aFace.y1 == aFace.y2)
                continue;

            // printf("%d,%d,%d,%d\n", aFace.x1, aFace.y1, aFace.x2, aFace.y2);
            // printf("%d,%d,%d,%d,%d,%d\n", aFace.blur, aFace.expression, aFace.illumination, aFace.invalid, aFace.occlusion, aFace.pose);
            // waitKey(500);
            if (aFace.invalid == 0)
                validFaces += 1;
            if (aFace.invalid == 1)
                invalidFaces += 1;

            anExample.faces.push_back(aFace);
        }
        examples.push_back(anExample);
    }
    printf("Valid faces  : %ld\n", validFaces);
    printf("Invalid faces: %ld\n", invalidFaces);

    /* NEURAL NETWORK CREATION - START */
    int inC = 30;
    int inR = 30;
    size_t nLayers = 15;
    size_t *neuronsPerLayer = new size_t[nLayers];
    nnInput = new double[inC * inR];
    char **fxs = new char *[nLayers];
    for (int l = 0; l < nLayers; l += 1)
    {
        neuronsPerLayer[l] = 500;
        if (l == 0)
            neuronsPerLayer[l] = 2 * inR * inC;
        if (l == nLayers - 1)
            neuronsPerLayer[l] = 7;

        fxs[l] = new char[15];
        strcpy(fxs[l], "fxLeakyReLU");
        if (l == nLayers - 1)
            strcpy(fxs[l], "fxSigmoid");
    }

    nn.createNeuronNetwork(inC * inR, nLayers, neuronsPerLayer);
    nn.setActivationFunction(&fxs);
    nn.setInputs(nnInput);

    size_t countWeights = nn.getWeightsNeeded();
    /* NEURAL NETWORK CREATION - END */

    /* PSO CREATION - START */
    size_t swarmDim = pso.createSwarm(15, countWeights);
    double *position = new double[swarmDim];
    double *velocity = new double[swarmDim];

    printf("Creating vectors %ld\n", swarmDim);
    cmHelper::Array::randomInit(swarmDim, position, -0.02, 0.02);
    cmHelper::Array::randomInit(swarmDim, velocity, -0.005, 0.005);
    printf("Creating vectors complete\n");

    pso.setFitnessFunction(trainNN);
    pso.initPosition(position, -2.0, 2.0);
    pso.initVelocity(velocity, -0.5, 0.5);
    pso.initWeights(0.729, 1.4944, 1.4944);
    /* PSO CREATION - END */

    for (int g = 0; g < 1000; g += 1)
    {
        pso.compute(15);
        printf("\r%4d - Best fitness: %.15lf (p: %ld)\n", g, pso.getBestFitness(), pso.getBestParticle());
        if (lastBest != pso.getBestFitness())
        {
            lastBest = pso.getBestFitness();
            char path[150];
            sprintf(path, "bests/%d.txt", g);
            nn.saveToFIle(path);
        }

        pso.evolve();

        // char key = (char)waitKey(1);
        // if (key == 27)
        //     break;
    }

    for (size_t l = 0; l < nLayers; l += 1)
        delete[] fxs[l];
    delete[] fxs;
    delete[] neuronsPerLayer;
    delete[] nnInput;
    delete[] position;
    delete[] velocity;

    return 0;
}