#include "cmActivationFunction.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

double cmNeuralNetwork::fxBinaryStep(double input)
{
    return input < 0.0 ? 0.0 : 1.0;
}

double cmNeuralNetwork::fxIdentity(double input)
{
    return input;
}

double cmNeuralNetwork::fxSigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double cmNeuralNetwork::fxTanh(double input)
{
    return tanh(input);
}

double cmNeuralNetwork::fxReLU(double input)
{
    return max(0.0, input);
}

double cmNeuralNetwork::fxLeakyReLU(double input)
{
    return max(0.1 * input, input);
}

double cmNeuralNetwork::fxELU(double input)
{
    double alpha = 0.1;
    return input >= 0.0 ? input : alpha * (exp(input) - 1.0);
}

double cmNeuralNetwork::fxGELU(double input)
{
    return (0.5 * input) * (1.0 + tanh(sqrt(2.0 / M_PI) * (input + (0.044715 * pow(input, 3)))));
}

double cmNeuralNetwork::fxSELU(double input)
{
    double alpha = 1.6732632423543772848170429916717;
    double lambda = 1.0507009873554804934193349852946;

    double out = input >= 0.0 ? input : alpha * (exp(input) - 1.0);
    return lambda * out;
}

double cmNeuralNetwork::fxSwish(double input)
{
    return input / (1.0 + exp(-input));
}
