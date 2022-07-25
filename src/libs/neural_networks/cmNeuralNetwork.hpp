#ifndef __CM_LIBS_DL_NEURAL_NETWORK__
#define __CM_LIBS_DL_NEURAL_NETWORK__

#include <vector>

using namespace std;

namespace cmNeuralNetwork
{
    class NNHelper
    {
    public:
        static void randomWeights(size_t n, double *output, double min = -0.05, double max = 0.05);
    };

    class Neuron
    {
    private:
        char _id[15];
        size_t _nI = 0;
        size_t _nW = 0;
        size_t _nE = 0;
        bool _isActive = true;
        long double *_inputs = NULL;
        long double *_extraInputs = NULL;
        long double _output = 0;
        double *_weights = NULL;

        long double (*_activationFunction)(long double) = NULL;

    public:
        Neuron(){};
        ~Neuron(){};
        bool isReady();

        void setID(size_t layerIndex, size_t neuronIndex);
        void setInputs(size_t n, long double *inputs);                               // Size and Pointer to the first element of inputs array.
        void setExtraInputs(size_t n, long double *inputs);                          // Size and Pointer to the first element of extra inputs array.
        void setWeights(size_t n, double *weights);                                  // Size and Pointer to the first element of weights array.
        void setActivationFunction(long double (*_activationFunction)(long double)); // Pointer to activation function.
        void setIsActive(bool isActive = true);

        size_t weightsNeeded();

        void printInputs();
        void printWights();
        void printInputSize();
        void printWeightSize();
        void printOutput();

        long double compute(bool softmax = false);
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
        long double *_output = NULL;

        long double *_inputs = NULL;
        long double *_extraInputs = NULL;
        double *_weights = NULL;

        void releaseMemory();

    public:
        Layer(){};
        ~Layer();
        bool isReady();

        void createLayer(size_t layerIndex, size_t numOfNeurons);
        void setInputs(size_t n, long double *inputs);
        void setWeights(size_t n, double *weights);
        void setActivationFunction(long double (*activationFunction)(long double));
        void setExtraInputs(size_t nExtraInputs, long double *extraInputs);

        size_t weightsNeeded();

        void printOutput();

        size_t getOutputSize();
        long double *getOtput();
        long double *compute(bool softMax = false);
    };
};

#endif