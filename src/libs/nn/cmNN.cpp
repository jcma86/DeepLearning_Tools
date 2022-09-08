#include "cmNN.hpp"

#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cmNN;

void Neuron::createNeuron(size_t index, size_t nInputs) {
  _id = index;
  _nI = nInputs;
  _nW = nInputs + 3;
}

bool Neuron::isReady() {
  bool ready = _nI > 0 && (_nI + _nE + 3) == _nW && _inputs != NULL &&
               _weights != NULL && _activationFunction != NULL;

  // printf("Here: %d %d %d %d %d\n", _nI > 0, (_nI + _nE + 1) == _nW, _inputs
  // != NULL, _weights != NULL, _activationFunction != NULL);

  if (!ready)
    printf(
        "[WARNING]: Neuron is not ready, please set inputs, weights (size of "
        "inputs and weights must be the same), bias and activation "
        "function.\n");

  return ready;
}

bool Neuron::isActive() {
  return _isActive;
}

void Neuron::setInputs(double* inputs) {
  _inputs = inputs;
}

void Neuron::setWeightsRange(double min, double max) {
  _minW = min;
  _maxW = max;
}

void Neuron::setWeights(double* weights, bool isLastLayer) {
  _isLastLayer = isLastLayer;
  _isActive = weights[_nW - 2] > 0.0 || _isLastLayer ? true : false;
  _weights = weights;

  setActivationFunction(_weights[_nW - 1]);
}

void Neuron::setExtraInputs(size_t n, double* inputs) {
  _nE = n;
  _extraInputs = inputs;
}

void Neuron::setActivationFunction(double w) {
  setActivationFunction(cmNN::getActivationFunctionName(w, _minW, _maxW));
}

void Neuron::setActivationFunction(double (*activationFunction)(double),
                                   const char* name) {
  strcpy(_actFxName, name);
  _activationFunction = activationFunction;
}

void Neuron::setActivationFunction(const char* fxName) {
  strcpy(_actFxName, fxName);
  _activationFunction = getActivationFunction(fxName);
}

void Neuron::setActivationFunction(NN_ACTIVATION_FX fxName) {
  strcpy(_actFxName, getActivationFunctionName(fxName));
  _activationFunction = getActivationFunction(fxName);
}

void Neuron::setIsActive(bool isActive) {
  _isActive = isActive;
}

size_t Neuron::weightsNeeded() {
  return (_nI + _nE + 3);  // bias and isActive that's why +2
}

double Neuron::compute(bool softmax) {
  if (!isReady() || !isActive())
    return 0.0;

  _output = 0.0;
  size_t i = 0;
  for (i = 0; i < _nI; i += 1)
    _output += (_inputs[i] * _weights[i]);
  for (size_t j = 0; j < _nE; j += 1, i += 1)
    _output += (_extraInputs[j] * _weights[i]);

  _output += _weights[_nW - 3];
  _output = _activationFunction(_output);

  if (softmax)
    _output = exp(_output);

  return _output;
}

// Layer
void Layer::releaseMemory() {
  if (_neuron)
    delete[] _neuron;

  if (_output)
    delete[] _output;

  _neuron = NULL;
  _output = NULL;
}

Layer::~Layer() {
  releaseMemory();
}

void Layer::createLayer(size_t layerIndex, size_t inSize, size_t numOfNeurons) {
  releaseMemory();

  _id = layerIndex;
  if (numOfNeurons == 0)
    printf("[WARNING]: Layer has 0 neurons.\n");

  _n = numOfNeurons;
  _nI = inSize;
  _neuron = new Neuron[numOfNeurons];
  _output = new double[numOfNeurons];

  if (!_neuron || !_output) {
    _n = 0;
    releaseMemory();
    printf("[ERROR]: Cannont create layer.\n");
    return;
  }

  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].createNeuron(i, _nI);
}

void Layer::setInputs(double* inputs) {
  _inputs = inputs;
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setInputs(_inputs);
}

void Layer::setWeights(double* weights) {
  _weights = weights;
  size_t wpn = _nI + _nE + 3;  // bias and isActivee that's why +2;
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setWeights(&_weights[i * wpn]);
}

void Layer::setWeightsRange(double min, double max) {
  for (size_t n = 0; n < _n; n += 1)
    _neuron[n].setWeightsRange(min, max);
}

void Layer::setExtraInputs(size_t nExtraInputs, double* extraInputs) {
  _nE = nExtraInputs;
  _extraInputs = extraInputs;
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setExtraInputs(_nE, _extraInputs);
}

void Layer::setActivationFunction(double (*activationFunction)(double),
                                  const char* fxName) {
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setActivationFunction(activationFunction, fxName);
}

void Layer::setActivationFunction(const char* fxName) {
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setActivationFunction(fxName);
}

void Layer::setActivationFunction(NN_ACTIVATION_FX fxName) {
  for (size_t i = 0; i < _n; i += 1)
    _neuron[i].setActivationFunction(fxName);
}

void Layer::setNeuronActivationFunction(size_t neuron, NN_ACTIVATION_FX fx) {
  _neuron[neuron].setActivationFunction(fx);
}

void Layer::setNeuronActivationFunction(size_t neuron, const char* fx) {
  _neuron[neuron].setActivationFunction(fx);
}

size_t Layer::weightsNeeded() {
  size_t wn = 0;
  for (size_t i = 0; i < _n; i += 1)
    wn += _neuron[i].weightsNeeded();

  return wn;
}

double* Layer::compute(Normalization norm, bool softMax) {
  double add = 0.0;
  double min = MAXFLOAT;
  double max = -MAXFLOAT;

  size_t activeNeurons = 0;
  for (size_t i = 0; i < _n; i += 1) {
    if (!_neuron[i].isActive()) {
      _output[i] = 0.0;
      continue;
    }

    _output[i] = _neuron[i].compute(softMax);
    min = _output[i] < min ? _output[i] : min;
    max = _output[i] > max ? _output[i] : max;
    add += _output[i];
    activeNeurons += 1;
  }

  if (norm == MIN_MAX && _n > 1) {
    double diff = max - min;
    for (size_t i = 0; i < _n; i += 1)
      _output[i] = _neuron[i].isActive() ? (_output[i] - min) / diff : 0.0;
  }

  if (norm == Z_SCORE) {
    double mean = add / activeNeurons;
    double sum = 0.0;
    for (size_t i = 0; i < _n; i += 1) {
      if (!_neuron[i].isActive())
        continue;
      sum += (_output[i] - mean) * (_output[i] - mean);
    }

    double standardDeviation = sqrt(sum / mean);
    for (size_t i = 0; i < _n; i += 1) {
      if (!_neuron[i].isActive())
        continue;
      _output[i] = (_output[i] - mean) / standardDeviation;
    }
  }

  if (softMax)
    for (size_t i = 0; i < _n; i += 1)
      _output[i] = _output[i] / add;

  return _output;
}

size_t Layer::getOutputSize() {
  return _n;
}

double* Layer::getOtput() {
  return _output;
}

// Neural Network
NeuralNetwork::~NeuralNetwork() {
  if (_layer) {
    delete[] _layer;
    _layer = NULL;
  }
}
void NeuralNetwork::createNeuronNetwork(size_t inSize,
                                        size_t nLayers,
                                        size_t* neuronsPerLayer) {
  _nLayers = nLayers;
  _nI = inSize;
  _neuronsPerLayer = neuronsPerLayer;

  _layer = new Layer[nLayers];
  _nW = 0;
  for (size_t l = 0; l < _nLayers; l += 1) {
    _layer[l].createLayer(l, l == 0 ? inSize : neuronsPerLayer[l - 1],
                          neuronsPerLayer[l]);
    _nW += _layer[l].weightsNeeded();
  }

  _output = _layer[_nLayers - 1].getOtput();
}

void NeuralNetwork::setInputs(double* inputs, bool onlyFirstLayer) {
  _inputs = inputs;

  _layer[0].setInputs(_inputs);
  if (!onlyFirstLayer) {
    for (size_t l = 1; l < _nLayers; l += 1)
      _layer[l].setInputs(_layer[l - 1].getOtput());
  }
}

void NeuralNetwork::setWeights(double* weights) {
  _weights = weights;

  size_t firstW = 0;
  for (size_t l = 0; l < _nLayers; l += 1) {
    size_t wn = _layer[l].weightsNeeded();
    _layer[l].setWeights(&_weights[firstW]);
    firstW += wn;
  }
}

void NeuralNetwork::setWeightsRange(double min, double max) {
  _minW = min;
  _maxW = max;
  for (size_t l = 0; l < _nLayers; l += 1)
    _layer[l].setWeightsRange(min, max);
}

void NeuralNetwork::setActivationFunction(char*** fxNames) {
  for (size_t l = 0; l < _nLayers; l += 1) {
    _layer[l].setActivationFunction((*fxNames)[l]);
  }
}

void NeuralNetwork::setLayerActivationFunction(size_t layer, const char* fx) {
  _layer[layer].setActivationFunction(fx);
}

void NeuralNetwork::setLayerActivationFunction(size_t layer,
                                               NN_ACTIVATION_FX fx) {
  _layer[layer].setActivationFunction(fx);
}

void NeuralNetwork::setNeuronActivationFunction(size_t layer,
                                                size_t neuron,
                                                NN_ACTIVATION_FX fx) {
  _layer[layer].setNeuronActivationFunction(neuron, fx);
}

void NeuralNetwork::setNeuronActivationFunction(size_t layer,
                                                size_t neuron,
                                                const char* fx) {
  _layer[layer].setNeuronActivationFunction(neuron, fx);
}

size_t NeuralNetwork::getWeightsNeeded() {
  return _nW;
}

double* NeuralNetwork::compute(Normalization norm, bool softMax) {
  // printf("Inp:%p\n", _inputs);
  for (size_t l = 0; l < _nLayers; l += 1)
    _layer[l].compute(l == _nLayers - 1 ? MIN_MAX : norm, softMax);

  return _output;
}

double* NeuralNetwork::getOutput() {
  return _output;
}

size_t NeuralNetwork::getInputSize() {
  return _nI;
}

size_t NeuralNetwork::getOutputSize() {
  return _layer[_nLayers - 1].getOutputSize();
}

void NeuralNetwork::printNeuralNetwork() {
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

void NeuralNetwork::saveToFile(const char* path, double* weights) {
  double* wToSave = weights ? weights : _weights;
  ofstream file(path);
  file << "++++ NeuralNetwork ++++\n";
  file << "LAYERS:" << _nLayers << "\n";
  file << "LAYER_SIZES:";
  for (size_t l = 0; l < _nLayers; l += 1)
    file << setprecision(15) << _neuronsPerLayer[l]
         << ((l == _nLayers - 1) ? "\n" : ",");
  file << "NUM_INPUTS:" << _nI << "\n";
  file << "MIN_THRESHOLD:0.9\n";
  file << "SOFT_MAX:0\n";
  file << "MIN_WEIGHT:" << _minW << "\n";
  file << "MAX_WEIGHT:" << _maxW << "\n";
  file << "NUM_WEIGHTS:" << _nW << "\n";
  file << "WEIGHTS:\n";
  for (size_t w = 0; w < _nW; w += 1) {
    char val[50];
    snprintf(val, 40, "%.15lf\n", wToSave[w]);
    file << val;
  }

  file.close();
}

void NeuralNetwork::saveToFile(const char* path,
                               double* weights,
                               NeuralNetworkConfiguration* config) {
  ofstream file(path);
  file << "++++ NeuralNetwork ++++\n";
  file << "LAYERS:" << config->nLayers << "\n";
  file << "LAYER_SIZES:";
  for (size_t l = 0; l < config->nLayers; l += 1)
    file << setprecision(15) << config->neuronsPerLayer[l]
         << ((l == config->nLayers - 1) ? "\n" : ",");
  file << "NUM_INPUTS:" << config->nInputs << "\n";
  file << "MIN_THRESHOLD:" << config->minThreshold << "\n";
  file << "SOFT_MAX:" << config->softMax << "\n";
  file << "MIN_WEIGHT:" << config->minW << "\n";
  file << "MAX_WEIGHT:" << config->maxW << "\n";
  file << "NUM_WEIGHTS:" << config->nWeights << "\n";
  file << "WEIGHTS:\n";
  for (size_t w = 0; w < config->nWeights; w += 1) {
    char val[50];
    snprintf(val, 40, "%.15lf\n", weights[w]);
    file << val;
  }

  file.close();
}

size_t NeuralNetwork::calculateNumberOfWeights(
    NeuralNetworkConfiguration* config) {
  size_t w = 0;
  for (size_t l = 0; l < config->nLayers; l += 1) {
    if (l == 0)
      w += (config->nInputs + 3) * config->neuronsPerLayer[l];
    else
      w += (config->neuronsPerLayer[l - 1] + 3) * config->neuronsPerLayer[l];
  }

  return w;
}

void NeuralNetwork::loadConfiguration(const char* filePath,
                                      NeuralNetworkConfiguration* configOutput,
                                      size_t nInputs) {
  fstream file;
  file.open(filePath, ios::in);

  if (!file.is_open()) {
    printf("[ERROR]: Failed to open file %s\n", filePath);
    return;
  }

  printf("Loading NN configuration from file.\n");

  configOutput->neuronsPerLayer = NULL;
  configOutput->nExtraInputs = 0;
  configOutput->nInputs = nInputs;
  configOutput->minThreshold = 0.9;
  configOutput->nLayers = 0;
  configOutput->nWeights = 0;
  configOutput->minW = -1.0;
  configOutput->maxW = 1.0;
  configOutput->weights = NULL;
  configOutput->softMax = false;

  char tokenProp = ':';
  char tokenVal = ',';
  string line;
  size_t count = 0;
  while (getline(file, line)) {
    size_t tokenPos = line.find(tokenProp);
    if (tokenPos != string::npos) {
      string prop = line.substr(0, tokenPos);
      line.erase(0, tokenPos + 1);
      if (prop.compare("LAYERS") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->nLayers;
        configOutput->neuronsPerLayer = new size_t[configOutput->nLayers];
      } else if (prop.compare("LAYER_SIZES") == 0) {
        for (size_t l = 0; l < configOutput->nLayers; l += 1) {
          tokenPos = line.find(tokenVal);
          string val = line.substr(0, tokenPos);
          line.erase(0, tokenPos + 1);
          stringstream sstream(val);
          sstream >> configOutput->neuronsPerLayer[l];
        }
      } else if (prop.compare("NUM_INPUTS") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->nInputs;
      } else if (prop.compare("MIN_THRESHOLD") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->minThreshold;
      } else if (prop.compare("SOFT_MAX") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->softMax;
      } else if (prop.compare("MIN_WEIGHT") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->minW;
      } else if (prop.compare("MAX_WEIGHT") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->maxW;
      } else if (prop.compare("NUM_WEIGHTS") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->nWeights;
        configOutput->weights = new double[configOutput->nWeights];
      } else if (prop.compare("WEIGHTS") == 0) {
        size_t cW = 0;
        while (getline(file, line) && cW < configOutput->nWeights) {
          stringstream sstream(line);
          sstream >> configOutput->weights[cW];
          cW += 1;
        }
      }
    }
  }
  if (configOutput->nWeights == 0) {
    configOutput->nWeights =
        (configOutput->nInputs + 3) * configOutput->neuronsPerLayer[0];
    for (size_t l = 1; l < configOutput->nLayers; l += 1)
      configOutput->nWeights += (configOutput->neuronsPerLayer[l - 1] + 3) *
                                (configOutput->neuronsPerLayer[l]);
  }
  printf("NN configuration loaded.\n");

  file.close();
}