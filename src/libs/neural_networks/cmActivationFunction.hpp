#ifndef __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__
#define __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__

namespace cmNeuralNetwork {
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
}

#endif