#ifndef __CM_LIBS_DL_NEURAL_NETWORK__
#define __CM_LIBS_DL_NEURAL_NETWORK__

#include <vector>

using namespace std;

namespace cmNeuralNetwork
{
    class NNHelper
    {
    public:
        static double *randomWeights(size_t n, double *output, double min = -1.0, double max = 1.0);
    };

    class Neuron
    {
    private:
        size_t _nI = 0;
        size_t _nW = 0;
        size_t _nE = 0;
        bool _isActive = true;
        double *_inputs = NULL;
        double *_extraInputs = NULL;
        double *_weights = NULL;
        double _output = 0;

        double (*_activationFunction)(double) = NULL;

    public:
        Neuron(){};
        ~Neuron(){};
        bool isReady();

        void setInputs(size_t n, double *inputs);                          // Size and Pointer to the first element of inputs array.
        void setExtraInputs(size_t n, double *inputs);                     // Size and Pointer to the first element of extra inputs array.
        void setWeights(size_t n, double *weights);                        // Size and Pointer to the first element of weights array.
        void setActivationFunction(double (*_activationFunction)(double)); // Pointer to activation function.
        void setIsActive(bool isActive = true);

        size_t weightsNeeded();

        void printInputs();
        void printWights();
        void printInputSize();
        void printWeightSize();
        void printOutput();

        double compute();
    };

    class Layer
    {
    private:
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

        void createLayer(size_t numOfNeurons);
        void setInputs(size_t n, double *inputs);
        void setWeights(size_t n, double *weights);
        void setActivationFunction(double (*activationFunction)(double));
        void setExtraInputs(size_t nExtraInputs, double *extraInputs);

        size_t weightsNeeded();

        void printOutput();

        size_t getOutputSize();
        double *getOtput();
        double *compute();
    };
};

#endif