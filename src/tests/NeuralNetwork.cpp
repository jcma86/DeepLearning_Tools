#include <stdlib.h>

#include "../libs/neural_networks/cmNeuralNetwork.hpp"
#include "../libs/neural_networks/cmActivationFunction.hpp"

using namespace cmNeuralNetwork;

int main()
{
    Neuron aNeuron;
    Layer aLayer;

    double in[2] = {1.1, 2.2};
    double w[3] = {8.4, 6.3, -1.0};

    aNeuron.setActivationFunction(fxSigmoid);
    aNeuron.setInputs(2, in);
    aNeuron.setWeights(2, w);

    aNeuron.compute();
    aNeuron.printInputs();
    aNeuron.printWights();
    aNeuron.printOutput();

    aLayer.createLayer(1);
    aLayer.initNeurons(2, in, w, fxSigmoid);
    aLayer.compute();
    aLayer.printOutput();

    printf("\n");

    return 0;
}
