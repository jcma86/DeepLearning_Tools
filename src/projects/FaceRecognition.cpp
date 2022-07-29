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
    unsigned int column;
    unsigned int row;
    unsigned int width;
    unsigned int height;

    unsigned char blur;
    unsigned char expression;
    unsigned char illumination;
    unsigned char occlusion;
    unsigned char pose;
    unsigned char invalid;
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

void buildExpectedResult(ExampleInfo example)
{
    Mat imgExample = imread("/Users/jose/Downloads/WIDER_DATASET/training/" + example.path, 0);

    for (size_t f = 0; f < example.faces.size(); f += 1)
    {
        circle(imgExample, Point2d(example.faces[f].column + example.faces[f].width / 2, example.faces[f].row + example.faces[f].height / 2), 5, 250, 2);
    }

    // rectangle(imgExample, Point2d(example.faces[0].column, example.faces[0].row), Point2d(example.faces[0].column + example.faces[0].width, example.faces[0].row + example.faces[0].height), 0, 3);
    imshow("Example", imgExample);

    waitKey(0);
}

// TODO: How to present inputs to NN
double trainNN(size_t n, double *weights)
{
    return 0.0;
}

int main()
{
    fstream file;
    file.open("/Users/jose/Downloads/WIDER_DATASET/wider_face_split/wider_face_train_bbx_gt.txt", ios::in);

    if (!file.is_open())
        return -1;

    string line;
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
            lineToParse >> aFace.column >> aFace.row >> aFace.width >> aFace.height;
            lineToParse >> aFace.blur >> aFace.expression >> aFace.illumination >> aFace.invalid >> aFace.occlusion >> aFace.pose;

            anExample.faces.push_back(aFace);
        }

        examples.push_back(anExample);
    }

    /* NEURAL NETWORK CREATION - START */
    int inC = 30;
    int inR = 30;
    size_t nLayers = 15;
    size_t *neuronsPerLayer = new size_t[nLayers];
    double *nnInput = new double[inC * inR];
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
    pso.initPosition(position, -1.0, 1.0);
    pso.initVelocity(velocity, -0.05, 0.05);
    pso.initWeights(0.729, 1.4944, 1.4944);
    /* PSO CREATION - END */

    // for (int g = 0; g < 1000; g += 1)
    // {
    //     pso.compute();
    //     printf("%4d - Best fitness: %d\n", g, (int)pso.getBestFitness());
    //     if (lastBest != pso.getBestFitness())
    //     {
    //         lastBest = pso.getBestFitness();
    //         char path[150];
    //         sprintf(path, "bests/%d.txt", (int)lastBest);
    //         nn.saveToFIle(path);
    //     }
    //     pso.evolve();

    //     char key = (char)waitKey(1);
    //     if (key == 27)
    //         break;
    // }

    for (size_t l = 0; l < nLayers; l += 1)
        delete[] fxs[l];
    delete[] fxs;
    delete[] neuronsPerLayer;
    delete[] nnInput;
    delete[] position;
    delete[] velocity;

    return 0;
}