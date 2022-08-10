#ifndef __CM_LIBS_DL_NEURAL_NETWORK__
#define __CM_LIBS_DL_NEURAL_NETWORK__

#include <stdio.h>
#include <math.h>
#include "cmActivationFunction.hpp"

using namespace std;

namespace cmNN
{
    typedef enum
    {
        NONE,
        MIN_MAX,
        Z_SCORE
    } Normalization;

    typedef struct
    {
        size_t nInputs;
        size_t nExtraInputs;
        size_t nWeights;

        size_t nLayers;
        size_t *neuronsPerLayer;

        double *weights;
    } NeuralNetworkConfiguration;

    class Neuron
    {
    private:
        char _id[15];
        size_t _nI = 0;
        size_t _nW = 0;
        size_t _nE = 0;
        bool _isActive = true;
        double *_inputs = NULL;
        double *_extraInputs = NULL;
        double _output = 0;
        double *_weights = NULL;

        double (*_activationFunction)(double) = NULL;
        char _actFxName[50];

    public:
        Neuron(){};
        ~Neuron(){};
        bool isReady();

        void createNeuron(size_t index, size_t nInputs);
        void setInputs(double *inputs);                                                        // Size and Pointer to the first element of inputs array.
        void setExtraInputs(size_t n, double *inputs);                                         // Size and Pointer to the first element of extra inputs array.
        void setWeights(double *weights);                                                      // Size and Pointer to the first element of weights array.
        void setActivationFunction(double (*_activationFunction)(double), const char *fxName); // Pointer to activation function.
        void setActivationFunction(const char *fxName);                                        // Pointer to activation function.
        void setActivationFunction(NN_ACTIVATION_FX fxName);                                   // Pointer to activation function.
        void setIsActive(bool isActive = true);

        size_t weightsNeeded();

        double compute(bool softmax = false);
    };

    class Layer
    {
    private:
        char _id[15];
        size_t _n = 0;
        size_t _nE = 0;
        size_t _nI = 0;
        size_t _nW = 0;

        Neuron *_neuron = NULL;
        double *_output = NULL;

        double *_inputs = NULL;
        double *_extraInputs = NULL;
        double *_weights = NULL;

        void releaseMemory();

    public:
        Layer(){};
        ~Layer();
        bool isReady();

        void createLayer(size_t layerIndex, size_t inSize, size_t numOfNeurons);
        void setInputs(double *inputs);
        void setWeights(double *weights);
        void setActivationFunction(double (*activationFunction)(double), const char *fxName);
        void setActivationFunction(const char *fxName);
        void setActivationFunction(NN_ACTIVATION_FX fxName);
        void setExtraInputs(size_t nExtraInputs, double *extraInputs);

        size_t weightsNeeded();

        size_t getOutputSize();
        double *getOtput();
        double *compute(Normalization norm = NONE, bool softMax = false);
    };

    class NeuralNetwork
    {
    private:
        size_t _nI = 0;
        size_t _nE = 0;
        size_t _nW = 0;

        size_t _nLayers = 0;
        size_t *_neuronsPerLayer;

        double *_inputs = NULL;
        double *_extraInputs = NULL;
        double *_weights = NULL;

        double *_output = NULL;

        Layer *_layer = NULL;

    public:
        NeuralNetwork(){};
        ~NeuralNetwork();
        void createNeuronNetwork(size_t inSize, size_t nLayers, size_t *neuronsPerLayer);
        void setInputs(double *inputs, bool onlyFirstLayer = false);
        void setWeights(double *weights);
        void setActivationFunction(char ***fxNames);

        size_t getWeightsNeeded();

        double *compute(Normalization norm = NONE, bool softMax = false);
        double *getOutput();
        size_t getInputSize();
        size_t getOutputSize();

        void printNeuralNetwork();
        void saveToFile(const char *path, double *weights = NULL);
        static void loadConfiguration(const char *filePath, NeuralNetworkConfiguration *configOutput);
        static void saveToFile(const char *path, double *weights, NeuralNetworkConfiguration *config);
    };
};

#endif