#include "cmActivationFunction.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

long double cmNeuralNetwork::fxBinaryStep(long double input)
{
    return input < 0.0 ? 0.0 : 1.0;
}

long double cmNeuralNetwork::fxIdentity(long double input)
{
    return input;
}

long double cmNeuralNetwork::fxSigmoid(long double input)
{
    return 1.0 / (1.0 + exp(-input));
}

long double cmNeuralNetwork::fxTanh(long double input)
{
    return tanh(input);
}

long double cmNeuralNetwork::fxReLU(long double input)
{
    return max((long double)0.0, input);
}

long double cmNeuralNetwork::fxLeakyReLU(long double input)
{
    return max(0.1 * input, input);
}

long double cmNeuralNetwork::fxELU(long double input)
{
    long double alpha = 0.1;
    return input >= 0.0 ? input : alpha * (exp(input) - 1.0);
}

long double cmNeuralNetwork::fxGELU(long double input)
{
    return (0.5 * input) * (1.0 + tanh(sqrt(2.0 / M_PI) * (input + (0.044715 * pow(input, 3)))));
}

long double cmNeuralNetwork::fxSELU(long double input)
{
    long double alpha = 1.6732632423543772848170429916717;
    long double lambda = 1.0507009873554804934193349852946;

    long double out = input >= 0.0 ? input : alpha * (exp(input) - 1.0);
    return lambda * out;
}

long double cmNeuralNetwork::fxSwish(long double input)
{
    return input / (1.0 + exp(-input));
}
