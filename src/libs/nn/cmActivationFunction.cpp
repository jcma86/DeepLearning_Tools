#include "cmActivationFunction.hpp"
#include <cmath>
#include <algorithm>
#include <string.h>

using namespace std;

const char *cmNN::getActivationFunctionName(NN_ACTIVATION_FX fx)
{
    if (fx == FX_BINARY_STEP)
        return "fxBinaryStep";
    if (fx == FX_IDENTITY)
        return "fxIdentity";
    if (fx == FX_SIGMOID)
        return "fxSigmoid";
    if (fx == FX_TANH)
        return "fxTanh";
    if (fx == FX_RELU)
        return "fxReLU";
    if (fx == FX_LEAKY_RELU)
        return "fxLeakyReLU";
    if (fx == FX_ELU)
        return "fxELU";
    if (fx == FX_GELU)
        return "fxGELU";
    if (fx == FX_SELU)
        return "fxSELU";
    if (fx == FX_SWISH)
        return "fxSwish";

    return "UndefinedFunction";
}

activationFx cmNN::getActivationFunction(NN_ACTIVATION_FX fx)
{
    if (fx == FX_BINARY_STEP)
        return fxBinaryStep;
    if (fx == FX_IDENTITY)
        return fxIdentity;
    if (fx == FX_SIGMOID)
        return fxSigmoid;
    if (fx == FX_TANH)
        return fxTanh;
    if (fx == FX_RELU)
        return fxReLU;
    if (fx == FX_LEAKY_RELU)
        return fxLeakyReLU;
    if (fx == FX_ELU)
        return fxELU;
    if (fx == FX_GELU)
        return fxGELU;
    if (fx == FX_SELU)
        return fxSELU;
    if (fx == FX_SWISH)
        return fxSwish;

    return fxUndefined;
}

activationFx cmNN::getActivationFunction(const char *fxName)
{
    NN_ACTIVATION_FX fx = FX_UNDEFINED;

    if (strcmp(fxName, "fxBinaryStep") == 0)
        fx = FX_BINARY_STEP;
    else if (strcmp(fxName, "fxIdentity") == 0)
        fx = FX_IDENTITY;
    else if (strcmp(fxName, "fxSigmoid") == 0)
        fx = FX_SIGMOID;
    else if (strcmp(fxName, "fxTanh") == 0)
        fx = FX_TANH;
    else if (strcmp(fxName, "fxReLU") == 0)
        fx = FX_RELU;
    else if (strcmp(fxName, "fxLeakyReLU") == 0)
        fx = FX_LEAKY_RELU;
    else if (strcmp(fxName, "fxELU") == 0)
        fx = FX_ELU;
    else if (strcmp(fxName, "fxGELU") == 0)
        fx = FX_GELU;
    else if (strcmp(fxName, "fxSELU") == 0)
        fx = FX_SELU;
    else if (strcmp(fxName, "fxSwish") == 0)
        fx = FX_SWISH;

    return getActivationFunction(fx);
}

double cmNN::fxUndefined(double input)
{
    printf("[WARNING]: Activation function not defined, using Identity instead.\n");
    return fxIdentity(input);
}

double cmNN::fxBinaryStep(double input)
{
    return input < 0.0 ? 0.0 : 1.0;
}

double cmNN::fxIdentity(double input)
{
    return input;
}

double cmNN::fxSigmoid(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double cmNN::fxTanh(double input)
{
    return tanh(input);
}

double cmNN::fxReLU(double input)
{
    return max((double)0.0, input);
}

double cmNN::fxLeakyReLU(double input)
{
    return max(0.1 * input, input);
}

double cmNN::fxELU(double input)
{
    double alpha = 0.1;
    return input >= 0.0 ? input : alpha * (exp(input) - 1.0);
}

double cmNN::fxGELU(double input)
{
    return (0.5 * input) * (1.0 + tanh(sqrt(2.0 / M_PI) * (input + (0.044715 * pow(input, 3)))));
}

double cmNN::fxSELU(double input)
{
    double alpha = 1.6732632423543772848170429916717;
    double lambda = 1.0507009873554804934193349852946;

    double out = input >= 0.0 ? input : alpha * (exp(input) - 1.0);
    return lambda * out;
}

double cmNN::fxSwish(double input)
{
    return input / (1.0 + exp(-input));
}
