#ifndef __CM_LIBS_DL_NEURAL_NETWORK__
#define __CM_LIBS_DL_NEURAL_NETWORK__

#include <math.h>
#include <stdio.h>
#include "cmActivationFunction.hpp"

using namespace std;

namespace cmNN {
typedef enum { NONE, MIN_MAX, Z_SCORE } Normalization;

typedef struct {
  size_t nInputs;
  size_t nExtraInputs;
  size_t nWeights;

  size_t nLayers;
  size_t* neuronsPerLayer;

  double* weights;
  double minW;
  double maxW;
  double minThreshold;
  bool softMax;
} NeuralNetworkConfiguration;

class Neuron {
 private:
  size_t _id;
  size_t _nI = 0;
  size_t _nW = 0;
  size_t _nE = 0;
  bool _isActive = true;
  bool _isLastLayer = false;
  bool _forceFx = false;
  double* _inputs = NULL;
  double* _extraInputs = NULL;
  double _output = 0;
  double* _weights = NULL;
  double _minW = 0.0;
  double _maxW = 0.0;

  double (*_activationFunction)(double) = NULL;
  char _actFxName[50];

 public:
  Neuron(){};
  ~Neuron(){};
  bool isReady();

  void createNeuron(size_t index, size_t nInputs);
  void setInputs(double* inputs);  // Size and Pointer to the first element of
                                   // inputs array.
  void setExtraInputs(size_t n,
                      double* inputs);  // Size and Pointer to the first element
                                        // of extra inputs array.
  void setWeights(double* weights,
                  bool isLastLayer = false);  // Size and Pointer to the first
                                              // element of weights array.

  void setWeightsRange(double min, double max);
  void setActivationFunction(
      double (*_activationFunction)(double),
      const char* fxName);  // Pointer to activation function.
  void setActivationFunction(
      const char* fxName);  // Pointer to activation function.
  void setActivationFunction(
      NN_ACTIVATION_FX fxName);  // Pointer to activation function.
  void setActivationFunction(double w);
  void setIsActive(bool isActive = true);
  bool isActive();

  size_t weightsNeeded();

  double compute(bool softmax = false);
};

class Layer {
 private:
  size_t _id;
  size_t _n = 0;
  size_t _nE = 0;
  size_t _nI = 0;
  size_t _nW = 0;

  Neuron* _neuron = NULL;
  double* _output = NULL;

  double* _inputs = NULL;
  double* _extraInputs = NULL;
  double* _weights = NULL;

  void releaseMemory();

 public:
  Layer(){};
  ~Layer();
  bool isReady();

  void createLayer(size_t layerIndex, size_t inSize, size_t numOfNeurons);
  void setInputs(double* inputs);
  void setWeights(double* weights);
  void setWeightsRange(double min, double max);
  void setActivationFunction(double (*activationFunction)(double),
                             const char* fxName);
  void setActivationFunction(const char* fxName);
  void setActivationFunction(NN_ACTIVATION_FX fxName);
  void setExtraInputs(size_t nExtraInputs, double* extraInputs);
  void setNeuronActivationFunction(size_t neuron, NN_ACTIVATION_FX fx);
  void setNeuronActivationFunction(size_t neuron, const char* fx);

  size_t weightsNeeded();

  size_t getOutputSize();
  double* getOtput();
  double* compute(Normalization norm = NONE, bool softMax = false);
};

class NeuralNetwork {
 private:
  size_t _nI = 0;
  size_t _nE = 0;
  size_t _nW = 0;

  size_t _nLayers = 0;
  size_t* _neuronsPerLayer;

  double* _inputs = NULL;
  double* _extraInputs = NULL;
  double* _weights = NULL;
  double _minW = 0.0;
  double _maxW = 0.0;

  double* _output = NULL;

  Layer* _layer = NULL;

 public:
  NeuralNetwork(){};
  ~NeuralNetwork();
  void createNeuronNetwork(size_t inSize,
                           size_t nLayers,
                           size_t* neuronsPerLayer);
  void setInputs(double* inputs, bool onlyFirstLayer = false);
  void setWeights(double* weights);
  void setWeightsRange(double min, double max);
  void setActivationFunction(char*** fxNames);

  void setLayerActivationFunction(size_t layer, const char* fx);
  void setLayerActivationFunction(size_t layer, NN_ACTIVATION_FX fx);
  void setNeuronActivationFunction(size_t layer, size_t neuron, const char* fx);
  void setNeuronActivationFunction(size_t layer,
                                   size_t neuron,
                                   NN_ACTIVATION_FX fx);

  size_t getWeightsNeeded();

  double* compute(Normalization norm = NONE, bool softMax = false);
  double* getOutput();
  size_t getInputSize();
  size_t getOutputSize();

  void printNeuralNetwork();
  void saveToFile(const char* path, double* weights = NULL);
  static void loadConfiguration(const char* filePath,
                                NeuralNetworkConfiguration* configOutput,
                                size_t nInputs = 0);
  static void saveToFile(const char* path,
                         double* weights,
                         NeuralNetworkConfiguration* config);

  static size_t calculateNumberOfWeights(NeuralNetworkConfiguration* config);
};
};  // namespace cmNN

#endif