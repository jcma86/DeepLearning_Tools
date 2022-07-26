#include "cmNeuralNetwork.hpp"

using namespace cmNeuralNetwork;

void Neuron::setID(size_t layerIndex, size_t neuronIndex)
{
    sprintf(_id, "l:%ld_n:%ld", layerIndex, neuronIndex);
}

bool Neuron::isReady()
{
    bool ready = _nI > 0 && (_nI + _nE + 1) == _nW && _inputs != NULL && _weights != NULL && _activationFunction != NULL;

    if (!ready)
        printf("[WARNING]: Neuron is not ready, please set inputs, weights (size of inputs and weights must be the same), bias and activation function.\n");

    return ready;
}

void Neuron::setInputs(size_t n, double *inputs)
{
    _nI = n;
    _inputs = inputs;
}

void Neuron::setWeights(size_t n, double *weights)
{
    _nW = n;
    _weights = weights;
}

void Neuron::setExtraInputs(size_t n, double *inputs)
{
    _nE = n;
    _extraInputs = inputs;
}

void Neuron::setActivationFunction(double (*activationFunction)(double))
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
        printf("%ld:%.15lf\n", i, _inputs[i]);
    for (size_t i = 0; i < _nE; i += 1)
        printf("extra-%ld:%.15lf\n", i, _extraInputs[i]);
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
    printf("%.15lf\n", _output);
}

double Neuron::compute(bool softmax)
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
    _output = new double[numOfNeurons];

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

void Layer::setInputs(size_t n, double *inputs)
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

void Layer::setExtraInputs(size_t nExtraInputs, double *extraInputs)
{
    _nE = nExtraInputs;
    _extraInputs = extraInputs;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setExtraInputs(_nE, _extraInputs);
}

void Layer::setActivationFunction(double (*activationFunction)(double))
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
        printf("%ld:%.15lf\n", i, _output[i]);
}

double *Layer::compute(Normalization norm, bool softMax)
{
    double add = 0.0;
    double min = MAXFLOAT;
    double max = -MAXFLOAT;

    for (size_t i = 0; i < _n; i += 1)
    {
        _output[i] = _neuron[i].compute(softMax);
        min = _output[i] < min ? _output[i] : min;
        max = _output[i] > max ? _output[i] : max;
        add += _output[i];
    }

    if (norm == MIN_MAX)
    {
        double diff = max - min;
        for (size_t i = 0; i < _n; i += 1)
            _output[i] = (_output[i] - min) / diff;
    }

    if (norm == Z_SCORE)
    {
        double mean = add / _n;
        double sum = 0.0;
        for (size_t i = 0; i < _n; i += 1)
            sum += (_output[i] - mean) * (_output[i] - mean);

        double standardDeviation = sqrt(sum / mean);
        for (size_t i = 0; i < _n; i += 1)
            _output[i] = (_output[i] - mean) / standardDeviation;
    }

    if (softMax)
        for (size_t i = 0; i < _n; i += 1)
            _output[i] = _output[i] / add;

    return _output;
}

size_t Layer::getOutputSize()
{
    return _n;
}

double *Layer::getOtput()
{
    return _output;
}