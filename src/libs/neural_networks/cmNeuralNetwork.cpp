#include "cmNeuralNetwork.hpp"
#include <random>

using namespace cmNeuralNetwork;

void NNHelper::randomWeights(size_t n, double *output, double min, double max)
{
    uniform_real_distribution<double> unif(min, max);
    random_device rd; // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd());

    for (size_t i = 0; i < n; i += 1)
        output[i] = unif(gen);
}

void Neuron::setID(size_t layerIndex, size_t neuronIndex){
    sprintf(_id, "l:%ld_n:%ld", layerIndex, neuronIndex);
}

bool Neuron::isReady()
{
    bool ready = _nI > 0 && (_nI + _nE + 1) == _nW && _inputs != NULL && _weights != NULL && _activationFunction != NULL;

    if (!ready)
        printf("[WARNING]: Neuron is not ready, please set inputs, weights (size of inputs and weights must be the same), bias and activation function.\n");

    return ready;
}

void Neuron::setInputs(size_t n, long double *inputs)
{
    _nI = n;
    _inputs = inputs;
}

void Neuron::setWeights(size_t n, double *weights)
{
    _nW = n;
    _weights = weights;
}

void Neuron::setExtraInputs(size_t n, long double *inputs)
{
    _nE = n;
    _extraInputs = inputs;
}

void Neuron::setActivationFunction(long double (*activationFunction)(long double))
{
    _activationFunction = activationFunction;
}

void Neuron::setIsActive(bool isActive)
{
    _isActive = isActive;
}

size_t Neuron::weightsNeeded()
{
    return (_nI + _nE + 1);
}

void Neuron::printInputs()
{
    if (_nI == 0 || _inputs == NULL)
    {
        printf("[WARNING]: Inputs not set, or size 0.\n");
        return;
    }

    for (size_t i = 0; i < _nI; i += 1)
        printf("%ld:%.15Lf\n", i, _inputs[i]);
    for (size_t i = 0; i < _nE; i += 1)
        printf("extra-%ld:%.15Lf\n", i, _extraInputs[i]);
}

void Neuron::printWights()
{
    if (_nW == 0 || _weights == NULL)
    {
        printf("[WARNING]: Weights not set, or size 0.\n");
        return;
    }

    for (size_t i = 0; i < _nW; i += 1)
        printf("%ld:%.15lf\n", i, _weights[i]);
}

void Neuron::printInputSize()
{
    printf("inputs:%ld\n", _nI);
}

void Neuron::printWeightSize()
{
    printf("weights:%ld\n", _nW);
}

void Neuron::printOutput()
{
    isReady();
    printf("%.15Lf\n", _output);
}

long double Neuron::compute(bool softmax)
{
    if (!isReady() || !_isActive)
        return 0.0;

    _output = 0.0;
    size_t i = 0;
    for (i = 0; i < _nI; i += 1)
        _output += (_inputs[i] * _weights[i]);
    for (size_t j = 0; j < _nE; j += 1, i += 1)
        _output += (_extraInputs[j] * _weights[i]);

    _output += _weights[_nW - 1];
    _output = _activationFunction(_output);

    if (softmax)
        _output = exp(_output);

    return _output;
}

// Layer
void Layer::releaseMemory()
{
    if (_neuron)
        delete[] _neuron;

    if (_output)
        delete[] _output;

    _neuron = NULL;
    _output = NULL;
}

Layer::~Layer()
{
    releaseMemory();
}

void Layer::createLayer(size_t layerIndex, size_t numOfNeurons)
{
    releaseMemory();

    sprintf(_id, "l:%ld", layerIndex);
    if (numOfNeurons == 0)
        printf("[WARNING]: Layer has 0 neurons.\n");

    _n = numOfNeurons;
    _neuron = new Neuron[numOfNeurons];
    _output = new long double[numOfNeurons];

    if (!_neuron || !_output)
    {
        _n = 0;
        releaseMemory();
        printf("[ERROR]: Cannont create layer.\n");
        return;
    }

    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setID(layerIndex, i);
}

void Layer::setInputs(size_t n, long double *inputs)
{
    if (n == 0)
        printf("[WARNING] Number of inputs equal to 0.\n");

    _nI = n;
    _inputs = inputs;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setInputs(_nI, _inputs);
}

void Layer::setWeights(size_t n, double *weights)
{
    if (n == 0)
        printf("[WARNING] Number of weights equal to 0.\n");

    _nW = n;
    _weights = weights;
    size_t wpn = _nI + _nE + 1;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setWeights(wpn, &_weights[i * wpn]);
}

void Layer::setExtraInputs(size_t nExtraInputs, long double *extraInputs)
{
    _nE = nExtraInputs;
    _extraInputs = extraInputs;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setExtraInputs(_nE, _extraInputs);
}

void Layer::setActivationFunction(long double (*activationFunction)(long double))
{
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setActivationFunction(activationFunction);
}

size_t Layer::weightsNeeded()
{
    size_t wn = 0;
    for (size_t i = 0; i < _n; i += 1)
        wn += _neuron[i].weightsNeeded();

    return wn;
}

void Layer::printOutput()
{
    for (size_t i = 0; i < _n; i += 1)
        printf("%ld:%.15Lf\n", i, _output[i]);
}

long double *Layer::compute(bool softMax)
{
    long double softMaxAdd = 0.0;
    for (size_t i = 0; i < _n; i += 1)
    {
        _output[i] = _neuron[i].compute(softMax);
        softMaxAdd += _output[i];
    }

    if (softMax)
        for (size_t i = 0; i < _n; i += 1)
            _output[i] = _output[i] / softMaxAdd;

    return _output;
}

size_t Layer::getOutputSize()
{
    return _n;
}

long double *Layer::getOtput()
{
    return _output;
}