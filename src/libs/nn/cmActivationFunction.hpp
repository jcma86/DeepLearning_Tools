#ifndef __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__
#define __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__

typedef double (*activationFx)(double);

namespace cmNN {
typedef enum {
  FX_UNDEFINED,
  FX_BINARY_STEP,
  FX_IDENTITY,
  FX_SIGMOID,
  FX_TANH,
  FX_RELU,
  FX_LEAKY_RELU,
  FX_ELU,
  FX_GELU,
  FX_SELU,
  FX_SWISH,
  COUNT
} NN_ACTIVATION_FX;

activationFx getActivationFunction(NN_ACTIVATION_FX fx);
activationFx getActivationFunction(const char* fxName);
const char* getActivationFunctionName(NN_ACTIVATION_FX fx);
const char* getActivationFunctionName(double w, double minW, double maxW);
int getActivationFunctionIndex(double w, double minW, double maxW);

double fxUndefined(double input);
double fxBinaryStep(double input);
double fxIdentity(double input);
double fxSigmoid(double input);
double fxTanh(double input);
double fxReLU(double input);
double fxLeakyReLU(double input);
double fxELU(double input);
double fxGELU(double input);
double fxSELU(double input);
double fxSwish(double input);
}  // namespace cmNN

#endif