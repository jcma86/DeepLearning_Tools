#ifndef __CM_LIBS_CNN__
#define __CM_LIBS_CNN__

#include <stdint.h>
#include <stdio.h>

using namespace std;

namespace cmCNN {
typedef struct {
  size_t w;
  size_t h;
  size_t d;
  size_t stride;
  size_t dilation;
} CNeuronDataSize;

typedef struct {
  size_t n;
  CNeuronDataSize inputSize;
  CNeuronDataSize kernelSize;
} CLayerSize;

typedef struct {
  CNeuronDataSize inputSize;
  size_t nParams;

  size_t nLayers;
  CLayerSize* layerConfig;

  double* params;
} CNeuralNetworkConfiguration;

class CNeuron {
 private:
  double* _input = NULL;
  char _activationFXName[50];

  CNeuronDataSize _inSize;
  CNeuronDataSize _kSize;

  size_t _dilation = 1;

  double* _kernel = NULL;
  size_t _stride = 1;
  uint8_t _maxPoolingSize = 1;

  double (*_activationFunction)(double) = NULL;

  double* _output = NULL;

  void releaseMemory();
  bool isReady();

 public:
  CNeuron() {}
  ~CNeuron();

  void createCNeuron(CNeuronDataSize inputSize,
                     CNeuronDataSize kernelSize,
                     size_t stride = 1,
                     size_t l = 1);
  void setInput(double* input);
  void setKernel(double* kernel);
  void setMaxPoolingSize(uint8_t maxPoolSize);
  void setActivationFunction(double (*_activationFunction)(double));
  void init();

  size_t getNumOfParamsNeeded();
  size_t getOutputWidth();
  size_t getOutputHeight();
  size_t getOutputDepth();
  double* getOutput();

  double* compute();
};

class CLayer {
 private:
  size_t _id;
  size_t _n;
  CNeuronDataSize _inSize;
  CNeuronDataSize _kSize;
  size_t _stride = 1;
  size_t _dilation = 1;
  size_t _paramsNeeded = 0;

  CNeuron* _cNeuron = NULL;
  double* _input = NULL;
  double* _kernels = NULL;
  double* _output = NULL;

  size_t _outD;
  size_t _outH;
  size_t _outW;

 public:
  CLayer(){};
  ~CLayer();
  void releaseMemory();
  void createCLayer(size_t layerIndex,
                    size_t numNeurons,
                    CNeuronDataSize _inSize,
                    CNeuronDataSize _kSize);
  void setInputs(double* input);
  void setKernels(double* kernels);
  double* compute();
  double* getOutput();

  size_t getNumOfParamsNeeded();
  size_t getOutputWidth();
  size_t getOutputHeight();
  size_t getOutputDepth();
};

class CNeuralNetwork {
 private:
  CNeuronDataSize _inSize;
  size_t _nLayers = 0;
  CLayer* _layer = NULL;

  double* _kernels = NULL;
  double* _input = NULL;

  void releaseMemory();

 public:
  CNeuralNetwork(){};
  ~CNeuralNetwork();
  void createCNeuronNetwork(CNeuralNetworkConfiguration* layerSizes);

  void setInputs(double* inputs, bool onlyFirstLayer = false);
  void setKernels(double* kernels);
  double* compute();
  double* getOuput();
  CNeuronDataSize getOutputSize();

  static CNeuronDataSize getLayerOutputSize(CLayerSize layerConfig);
  static void loadConfiguration(const char* filePath,
                                CNeuralNetworkConfiguration* configOutput);
};
}  // namespace cmCNN

#endif
