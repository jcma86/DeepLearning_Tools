#include "cmCNN.hpp"
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cmCNN;

CNeuronDataSize CNeuralNetwork::getLayerOutputSize(CLayerSize layerConfig) {
  CNeuronDataSize size;

  // Width
  size.w =
      ((layerConfig.inputSize.w -
        (layerConfig.kernelSize.w + ((layerConfig.kernelSize.w - 1) *
                                     (layerConfig.kernelSize.dilation - 1)))) /
       layerConfig.kernelSize.stride) +
      1;

  // Height
  size.h =
      ((layerConfig.inputSize.h -
        (layerConfig.kernelSize.h + ((layerConfig.kernelSize.h - 1) *
                                     (layerConfig.kernelSize.dilation - 1)))) /
       layerConfig.kernelSize.stride) +
      1;

  size.d = layerConfig.n;

  return size;
}

void CNeuralNetwork::loadConfiguration(
    const char* filePath,
    CNeuralNetworkConfiguration* configOutput) {
  fstream file;
  file.open(filePath, ios::in);

  if (!file.is_open()) {
    printf("[ERROR]: Failed to open file %s\n", filePath);
    return;
  }

  printf("Loading CNN configuration from file.\n");

  configOutput->inputSize.d = 0;
  configOutput->inputSize.w = 0;
  configOutput->inputSize.h = 0;

  configOutput->nLayers = 0;
  configOutput->nParams = 0;
  configOutput->params = NULL;

  char tokenProp = ':';
  char tokenVal = ',';
  char tokenDim = '&';
  string line;
  size_t count = 0;
  while (getline(file, line)) {
    size_t tokenPos = line.find(tokenProp);
    if (tokenPos != string::npos) {
      string prop = line.substr(0, tokenPos);
      line.erase(0, tokenPos + 1);
      if (prop.compare("INPUT_SIZE") == 0) {
        tokenPos = line.find(tokenDim);
        string val = line.substr(0, tokenPos);
        line.erase(0, tokenPos + 1);
        stringstream sstream(val);
        sstream >> configOutput->inputSize.w;

        tokenPos = line.find(tokenDim);
        val = line.substr(0, tokenPos);
        line.erase(0, tokenPos + 1);
        sstream = stringstream(val);
        sstream >> configOutput->inputSize.h;

        sstream = stringstream(line);
        sstream >> configOutput->inputSize.d;
      } else if (prop.compare("LAYERS") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->nLayers;
        configOutput->layerConfig = new CLayerSize[configOutput->nLayers];
      } else if (prop.compare("LAYER_SIZES") == 0) {
        for (size_t l = 0; l < configOutput->nLayers; l += 1) {
          tokenPos = line.find(tokenVal);
          string val = line.substr(0, tokenPos);
          line.erase(0, tokenPos + 1);

          tokenPos = val.find(tokenProp);
          string num = val.substr(0, tokenPos);
          val.erase(0, tokenPos + 1);
          stringstream sstream(num);
          sstream >> configOutput->layerConfig[l].n;

          tokenPos = val.find(tokenDim);
          num = val.substr(0, tokenPos);
          sstream = stringstream(num);
          sstream >> configOutput->layerConfig[l].kernelSize.w;
          val.erase(0, tokenPos + 1);

          tokenPos = val.find(tokenDim);
          num = val.substr(0, tokenPos);
          sstream = stringstream(num);
          sstream >> configOutput->layerConfig[l].kernelSize.h;
          val.erase(0, tokenPos + 1);

          tokenPos = val.find(tokenDim);
          num = val.substr(0, tokenPos);
          sstream = stringstream(num);
          sstream >> configOutput->layerConfig[l].kernelSize.stride;
          val.erase(0, tokenPos + 1);

          sstream = stringstream(val);
          sstream >> configOutput->layerConfig[l].kernelSize.dilation;
        }
      } else if (prop.compare("NUM_PARAMS") == 0) {
        stringstream sstream(line);
        sstream >> configOutput->nParams;
        configOutput->params = new double[configOutput->nParams];
      } else if (prop.compare("PARAMS") == 0) {
        size_t cW = 0;
        while (getline(file, line) && cW < configOutput->nParams) {
          stringstream sstream(line);
          sstream >> configOutput->params[cW];
          cW += 1;
        }
      }
      // else if (prop.compare("MIN_THRESHOLD") == 0) {
      //   stringstream sstream(line);
      //   sstream >> configOutput->minThreshold;
      // } else if (prop.compare("SOFT_MAX") == 0) {
      //   stringstream sstream(line);
      //   sstream >> configOutput->softMax;
      // } else if (prop.compare("MIN_WEIGHT") == 0) {
      //   stringstream sstream(line);
      //   sstream >> configOutput->minW;
      // } else if (prop.compare("MAX_WEIGHT") == 0) {
      //   stringstream sstream(line);
      //   sstream >> configOutput->maxW;
      // }
    }
  }

  size_t tparams = 0;
  for (size_t l = 0; l < configOutput->nLayers; l += 1) {
    if (l == 0) {
      configOutput->layerConfig[l].kernelSize.d = configOutput->inputSize.d;
      configOutput->layerConfig[l].inputSize.w = configOutput->inputSize.w;
      configOutput->layerConfig[l].inputSize.h = configOutput->inputSize.h;
      configOutput->layerConfig[l].inputSize.d = configOutput->inputSize.d;
    } else {
      CNeuronDataSize s = getLayerOutputSize(configOutput->layerConfig[l - 1]);

      configOutput->layerConfig[l].inputSize.w = s.w;
      configOutput->layerConfig[l].inputSize.h = s.h;
      configOutput->layerConfig[l].inputSize.d = s.d;
      configOutput->layerConfig[l].kernelSize.d = s.d;
    }

    tparams += configOutput->layerConfig[l].n *
               (configOutput->layerConfig[l].kernelSize.w *
                configOutput->layerConfig[l].kernelSize.h *
                configOutput->layerConfig[l].kernelSize.d);
  }

  // for (size_t l = 0; l < configOutput->nLayers; l += 1) {
  //   CNeuronDataSize s = getLayerOutput(configOutput->layerConfig[l]);
  //   printf("%ld,%ld,%ld  ---  %ld,%ld,%ld\n",
  //          configOutput->layerConfig[l].inputSize.d,
  //          configOutput->layerConfig[l].inputSize.w,
  //          configOutput->layerConfig[l].inputSize.h, s.d, s.w, s.h);
  // }

  printf("CNN configuration loaded.\n");

  file.close();
}

void CNeuralNetwork::saveToFile(const char* path,
                                double* params,
                                CNeuralNetworkConfiguration* config) {
  ofstream file(path);
  file << "++++ ConvolutionalNeuralNetwork ++++\n";
  file << "INPUT_SIZE:" << config->inputSize.w << "&" << config->inputSize.h
       << "&" << config->inputSize.d << "\n";
  file << "LAYERS:" << config->nLayers << "\n";
  file << "LAYER_SIZES:";
  for (size_t l = 0; l < config->nLayers; l += 1)
    file << config->layerConfig[l].n << ":"
         << config->layerConfig[l].kernelSize.w << "&"
         << config->layerConfig[l].kernelSize.h << "&"
         << config->layerConfig[l].kernelSize.stride << "&"
         << config->layerConfig[l].kernelSize.dilation
         << ((l == config->nLayers - 1) ? "\n" : ",");
  file << "NUM_PARAMS:" << config->nParams << "\n";
  file << "PARAMS:\n";
  for (size_t p = 0; p < config->nParams; p += 1) {
    char val[50];
    snprintf(val, 40, "%.15lf\n", params[p]);
    file << val;
  }

  file.close();
}

CNeuron::~CNeuron() {
  releaseMemory();
}

void CNeuron::releaseMemory() {
  if (_output)
    delete[] _output;

  _output = NULL;
}

bool CNeuron::isReady() {
  return _input && _kernel;
}

size_t CNeuron::getId() {
  return _id;
}

void CNeuron::createCNeuron(size_t index,
                            CNeuronDataSize inputSize,
                            CNeuronDataSize kernelSize,
                            size_t stride,
                            size_t dilation) {
  _id = index;
  _inSize.d = inputSize.d;
  _inSize.w = inputSize.w;
  _inSize.h = inputSize.h;

  _kSize.d = kernelSize.d;
  _kSize.w = kernelSize.w;
  _kSize.h = kernelSize.h;

  _stride = stride;
  _dilation = dilation;

  releaseMemory();
  _output = new double[getOutputDepth() * getOutputHeight() * getOutputWidth()];
}

void CNeuron::setInput(double* input) {
  _input = input;
}

void CNeuron::setKernel(double* kernel) {
  _kernel = kernel;
}

void CNeuron::setActivationFunction(double (*activationFunction)(double)) {
  _activationFunction = activationFunction;
}

double* CNeuron::getOutput() {
  return _output;
}

size_t CNeuron::getNumOfParamsNeeded() {
  return (_kSize.d * _kSize.w * _kSize.h);
}

CNeuronDataSize CNeuron::getInputSize() {
  return _inSize;
}

CNeuronDataSize CNeuron::getKernelSize() {
  return _kSize;
}

size_t CNeuron::getOutputWidth() {
  return ((_inSize.w - (_kSize.w + ((_kSize.w - 1) * (_dilation - 1)))) /
          _stride) +
         1;
}

size_t CNeuron::getOutputHeight() {
  return ((_inSize.h - (_kSize.h + ((_kSize.h - 1) * (_dilation - 1)))) /
          _stride) +
         1;
}

size_t CNeuron::getOutputDepth() {
  return 1;  //((_inD - (_kD + ((_kD + 1) * (_dilation - 1)))) / _stride) + 1;
}

double* CNeuron::compute() {
  if (!isReady()) {
    printf("No ready \n");
    return NULL;
  }

  if (_inSize.d != _kSize.d) {
    printf(
        "[WARNING]: Input and kernel must have same depth (neuron: %ld in: "
        "%ld, kernel: "
        "%ld)\n",
        _id, _inSize.d, _kSize.d);
    return NULL;
  }

  size_t wStart = (_kSize.w / 2);
  wStart += (wStart * (_dilation - 1));
  size_t hStart = (_kSize.h / 2);
  hStart += (hStart * (_dilation - 1));

  size_t outIndex = 0;
  for (int indexH = (int)hStart; indexH < (int)(_inSize.h - hStart);
       indexH += _stride) {
    for (int indexW = (int)wStart; indexW < (int)(_inSize.w - wStart);
         indexW += _stride) {
      _output[outIndex] = 0.0;
      for (int indexKD = 0; indexKD < (int)(_kSize.d); indexKD += _stride) {
        int lHPad = 0;
        for (int indexKH = -(int)hStart; indexKH < (int)(_kSize.h - hStart);
             indexKH += _stride) {
          int lWPad = 0;
          for (int indexKW = -(int)wStart; indexKW < (int)(_kSize.w - wStart);
               indexKW += _stride) {
            size_t inW = indexW + indexKW + lWPad;
            size_t inH = indexH + indexKH + lHPad;
            size_t kW = indexKW + wStart;
            size_t kH = indexKH + hStart;
            size_t inIndex =
                ((inH * _inSize.w) + inW) + (indexKD * (_inSize.w * _inSize.h));
            size_t kIndex =
                ((kH * _kSize.w) + kW) + (indexKD * (_kSize.w * _kSize.h));
            _output[outIndex] += (_kernel[kIndex] * _input[inIndex]);
            lWPad += (_dilation - 1);
          }
          lHPad += (_dilation - 1);
        }
      }
      outIndex += 1;
    }
  }

  return _output;
}

// CLayer
void CLayer::releaseMemory() {
  if (_cNeuron != NULL)
    delete[] _cNeuron;
  if (_output != NULL)
    delete[] _output;
}

CLayer::~CLayer() {
  releaseMemory();
}

void CLayer::createCLayer(size_t layerIndex,
                          size_t numNeurons,
                          CNeuronDataSize inputSize,
                          CNeuronDataSize kernelSize) {
  if (inputSize.d != kernelSize.d) {
    printf("[WARNING]: Input and Kernel should have same depth.\n");
    return;
  }

  releaseMemory();

  _id = layerIndex;
  _n = numNeurons;
  _inSize.d = inputSize.d;
  _inSize.w = inputSize.w;
  _inSize.h = inputSize.h;
  _kSize.d = kernelSize.d;
  _kSize.w = kernelSize.w;
  _kSize.h = kernelSize.h;

  _stride = kernelSize.stride;
  _dilation = kernelSize.dilation;

  _paramsNeeded = 0;
  _cNeuron = new CNeuron[numNeurons];
  for (size_t n = 0; n < numNeurons; n += 1) {
    _cNeuron[n].createCNeuron(n, _inSize, _kSize, _stride, _dilation);
    _paramsNeeded += _cNeuron[n].getNumOfParamsNeeded();
  }

  _outD = _cNeuron[0].getOutputDepth();
  _outW = _cNeuron[0].getOutputWidth();
  _outH = _cNeuron[0].getOutputHeight();

  _output = new double[(_outD * _outH * _outW * _n)];
}

void CLayer::setInputs(double* input) {
  _input = input;

  for (size_t n = 0; n < _n; n += 1)
    _cNeuron[n].setInput(_input);
}

void CLayer::setKernels(double* kernels) {
  _kernels = kernels;
  size_t offset = 0;
  for (size_t n = 0; n < _n; n += 1) {
    _cNeuron[n].setKernel(&_kernels[offset]);
    offset += _cNeuron[n].getNumOfParamsNeeded();
  }
}

double* CLayer::compute() {
  for (size_t n = 0; n < _n; n += 1) {
    double* out = _cNeuron[n].compute();
    size_t outSize =
        _cNeuron[n].getOutputWidth() * _cNeuron[n].getOutputHeight();
    for (size_t c = 0; c < outSize; c += 1)
      _output[(n * outSize) + c] = out[c];
  }

  return _output;
}

double* CLayer::getOutput() {
  return _output;
}

size_t CLayer::getNumOfParamsNeeded() {
  return _paramsNeeded;
}

size_t CLayer::getId() {
  return _id;
}

size_t CLayer::getNumOfCNeurons() {
  return _n;
}

CNeuron* CLayer::getCNeuron(size_t neuronId) {
  if (neuronId > _n) {
    printf("[WARNING] Neuron id out of range (%ld neurons).\n", _n);
    return NULL;
  }
  return &_cNeuron[neuronId];
}

size_t CLayer::getOutputWidth() {
  return _outW;
}

size_t CLayer::getOutputHeight() {
  return _outH;
}

size_t CLayer::getOutputDepth() {
  return _outD * _n;
}

// CNeuralNetwork
void CNeuralNetwork::releaseMemory() {
  if (_layer != NULL) {
    delete[] _layer;
    _layer = NULL;
  }
}

CNeuralNetwork::~CNeuralNetwork() {
  releaseMemory();
}

void CNeuralNetwork::createCNeuronNetwork(
    CNeuralNetworkConfiguration* cnnConfig) {
  releaseMemory();
  _inSize.d = cnnConfig->inputSize.d;
  _inSize.w = cnnConfig->inputSize.w;
  _inSize.h = cnnConfig->inputSize.h;
  _nLayers = cnnConfig->nLayers;

  _layer = new CLayer[_nLayers];
  for (size_t l = 0; l < _nLayers; l += 1) {
    _layer[l].createCLayer(l, cnnConfig->layerConfig[l].n,
                           cnnConfig->layerConfig[l].inputSize,
                           cnnConfig->layerConfig[l].kernelSize);
  }
}

void CNeuralNetwork::setInputs(double* inputs, bool onlyFirstLayer) {
  _layer[0].setInputs(inputs);
  if (!onlyFirstLayer) {
    for (size_t l = 1; l < _nLayers; l += 1)
      _layer[l].setInputs(_layer[l - 1].getOutput());
  }
}

void CNeuralNetwork::setKernels(double* kernels) {
  size_t offset = 0;

  for (size_t l = 0; l < _nLayers; l += 1) {
    _layer[l].setKernels(&kernels[offset]);
    offset += _layer[l].getNumOfParamsNeeded();
  }
}

double* CNeuralNetwork::compute() {
  for (size_t l = 0; l < _nLayers; l += 1)
    _layer[l].compute();

  return _layer[_nLayers - 1].getOutput();
}

CLayer* CNeuralNetwork::getCLayer(size_t layerId) {
  if (layerId > _nLayers) {
    printf("[WARNING] Layer id out of range (%ld layers).\n", _nLayers);
    return NULL;
  }

  return &_layer[layerId];
}

CNeuron* CNeuralNetwork::getCNeuron(size_t layerId, size_t neuronId) {
  CLayer* layer = getCLayer(layerId);
  if (!layer)
    return NULL;

  return layer->getCNeuron(neuronId);
}

double* CNeuralNetwork::getOuput() {
  return _layer[_nLayers - 1].getOutput();
}

size_t CNeuralNetwork::getNumOfParamsNeeded() {
  size_t nParams = 0;
  for (size_t l = 0; l < _nLayers; l += 1)
    nParams += _layer[l].getNumOfParamsNeeded();

  return nParams;
}

CNeuronDataSize CNeuralNetwork::getOutputSize() {
  CNeuronDataSize size;

  size.d = _layer[_nLayers - 1].getOutputDepth();
  size.w = _layer[_nLayers - 1].getOutputWidth();
  size.h = _layer[_nLayers - 1].getOutputHeight();

  size.dilation = 1;
  size.stride = 1;

  return size;
}
