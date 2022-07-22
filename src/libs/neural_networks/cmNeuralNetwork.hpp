#ifndef __CM_LIBS_DL_NEURAL_NETWORK__
#define __CM_LIBS_DL_NEURAL_NETWORK__

#include <vector>

using namespace std;

namespace cmNeuralNetwork
{
    class Neuron
    {
    private:
        size_t _nI = 0;
        size_t _nW = 0;
        double *_inputs = NULL;
        double *_weights = NULL;
        double _output = 0;

        double (*_activationFunction)(double) = NULL;

    public:
        Neuron(){};
        ~Neuron(){};
        bool isReady();

        void setInputs(size_t n, double *inputs);                          // Size and Pointer to the first element of inputs array.
        void setWeights(size_t n, double *weights);                        // Size and Pointer to the first element of weights array.
        void setActivationFunction(double (*_activationFunction)(double)); // Pointer to activation function.

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
        size_t _nI = 0;
        size_t _nW = 0;

        Neuron *_neuron = NULL;
        double *_output = NULL;

        double *_inputs = NULL;
        double *_weights = NULL;

        void releaseMemory();

    public:
        Layer(){};
        ~Layer();
        bool isReady();

        void createLayer(size_t numOfNeurons);
        void initNeurons(size_t n, double *inputs, double *weights, double (*activationFunction)(double) = NULL);

        void printOutput();

        double *getOtput();
        double *compute();
    };
};

#endif