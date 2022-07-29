#include "cmNN.hpp"

#include <string.h>
#include <iostream>
#include <fstream>

using namespace cmNN;

void Neuron::createNeuron(size_t index, size_t nInputs)
{
    sprintf(_id, "n:%ld", index);
    _nI = nInputs;
    _nW = nInputs + 1;
}

bool Neuron::isReady()
{
    bool ready = _nI > 0 && (_nI + _nE + 1) == _nW && _inputs != NULL && _weights != NULL && _activationFunction != NULL;

    if (!ready)
        printf("[WARNING]: Neuron is not ready, please set inputs, weights (size of inputs and weights must be the same), bias and activation function.\n");

    return ready;
}

void Neuron::setInputs(double *inputs)
{
    _inputs = inputs;
}

void Neuron::setWeights(double *weights)
{
    _weights = weights;
}

void Neuron::setExtraInputs(size_t n, double *inputs)
{
    _nE = n;
    _extraInputs = inputs;
}

void Neuron::setActivationFunction(double (*activationFunction)(double), const char *name)
{
    strcpy(_actFxName, name);
    _activationFunction = activationFunction;
}

void Neuron::setActivationFunction(const char *fxName)
{
    strcpy(_actFxName, fxName);
    _activationFunction = getActivationFunction(fxName);
}

void Neuron::setActivationFunction(NN_ACTIVATION_FX fxName)
{
    strcpy(_actFxName, getActivationFunctionName(fxName));
    _activationFunction = getActivationFunction(fxName);
}

void Neuron::setIsActive(bool isActive)
{
    _isActive = isActive;
}

size_t Neuron::weightsNeeded()
{
    return (_nI + _nE + 1);
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

void Layer::createLayer(size_t layerIndex, size_t inSize, size_t numOfNeurons)
{
    releaseMemory();

    sprintf(_id, "l:%ld", layerIndex);
    if (numOfNeurons == 0)
        printf("[WARNING]: Layer has 0 neurons.\n");

    _n = numOfNeurons;
    _nI = inSize;
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
        _neuron[i].createNeuron(i, _nI);
}

void Layer::setInputs(double *inputs)
{
    _inputs = inputs;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setInputs(_inputs);
}

void Layer::setWeights(double *weights)
{
    _weights = weights;
    size_t wpn = _nI + _nE + 1;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setWeights(&_weights[i * wpn]);
}

void Layer::setExtraInputs(size_t nExtraInputs, double *extraInputs)
{
    _nE = nExtraInputs;
    _extraInputs = extraInputs;
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setExtraInputs(_nE, _extraInputs);
}

void Layer::setActivationFunction(double (*activationFunction)(double), const char *fxName)
{
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setActivationFunction(activationFunction, fxName);
}

void Layer::setActivationFunction(const char *fxName)
{
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setActivationFunction(fxName);
}

void Layer::setActivationFunction(NN_ACTIVATION_FX fxName)
{
    for (size_t i = 0; i < _n; i += 1)
        _neuron[i].setActivationFunction(fxName);
}

size_t Layer::weightsNeeded()
{
    size_t wn = 0;
    for (size_t i = 0; i < _n; i += 1)
        wn += _neuron[i].weightsNeeded();

    return wn;
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

// Neural Network
NeuralNetwork::~NeuralNetwork()
{
    if (_layer)
    {
        delete[] _layer;
        _layer = NULL;
    }
}
void NeuralNetwork::createNeuronNetwork(size_t inSize, size_t nLayers, size_t *neuronsPerLayer)
{
    _nLayers = nLayers;
    _nI = inSize;
    _neuronsPerLayer = neuronsPerLayer;

    _layer = new Layer[nLayers];
    _nW = 0;
    for (size_t l = 0; l < _nLayers; l += 1)
    {
        _layer[l].createLayer(l, l == 0 ? inSize : neuronsPerLayer[l - 1], neuronsPerLayer[l]);
        _nW += _layer[l].weightsNeeded();
    }

    _output = _layer[_nLayers - 1].getOtput();
}

void NeuralNetwork::setInputs(double *inputs)
{
    _inputs = inputs;

    size_t firstIn = 0;
    for (size_t l = 0; l < _nLayers; l += 1)
        _layer[l].setInputs(l == 0 ? _inputs : _layer[l - 1].getOtput());
}

void NeuralNetwork::setWeights(double *weights)
{
    _weights = weights;

    size_t firstW = 0;
    for (size_t l = 0; l < _nLayers; l += 1)
    {
        size_t wn = _layer[l].weightsNeeded();
        _layer[l].setWeights(&_weights[firstW]);
        firstW += wn;
    }
}

void NeuralNetwork::setActivationFunction(char ***fxNames)
{
    for (size_t l = 0; l < _nLayers; l += 1){
        _layer[l].setActivationFunction((*fxNames)[l]);
    }
}

size_t NeuralNetwork::getWeightsNeeded()
{
    return _nW;
}

double *NeuralNetwork::compute(Normalization norm, bool softMax)
{
    for (size_t l = 0; l < _nLayers; l += 1)
        _layer[l].compute(l == _nLayers - 1 ? MIN_MAX : norm, softMax);

    return _output;
}

double *NeuralNetwork::getOutput()
{
    return _output;
}

size_t NeuralNetwork::getOutputSize()
{
    return _layer[_nLayers - 1].getOutputSize();
}

void NeuralNetwork::printNeuralNetwork()
{
    printf("++++ NeuralNetwork ++++\n");
    printf("LAYERS:%ld\n", _nLayers);
    printf("LAYER_SIZES:");
    for (size_t l = 0; l < _nLayers; l += 1)
        printf("%ld%s", _neuronsPerLayer[l], l == _nLayers - 1 ? "\n" : ",");
    printf("NUM_INPUTS:%ld\n", _nI);
    printf("NUM_WEIGHTS:%ld\n", _nW);
    printf("WEIGHTS:\n");
    for (size_t w = 0; w < _nW; w += 1)
        printf("%.15lf\n", _weights[w]);
}

void NeuralNetwork::saveToFIle(const char *path)
{
    ofstream file(path);
    file << "++++ NeuralNetwork ++++\n";
    file << "LAYERS:" << _nLayers << "\n";
    file << "LAYER_SIZES:";
    for (size_t l = 0; l < _nLayers; l += 1)
        file << setprecision(16) << _neuronsPerLayer[l] << ((l == _nLayers - 1) ? "\n" : ",");
    file << "NUM_INPUTS:" << _nI << "\n";
    file << "NUM_WEIGHTS:" << _nW << "\n";
    file << "WEIGHTS:\n";
    for (size_t w = 0; w < _nW; w += 1)
        file << setprecision(16) << _weights[w] << "\n";

    file.close();
}
