#ifndef __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__
#define __CM_LIBS_DL_NEURAL_NETWORK_ACTIVATION_FX__

namespace cmNeuralNetwork {
    long double fxBinaryStep(long double input);
    long double fxIdentity(long double input);
    long double fxSigmoid(long double input);
    long double fxTanh(long double input);
    long double fxReLU(long double input);
    long double fxLeakyReLU(long double input);
    long double fxELU(long double input);
    long double fxGELU(long double input);
    long double fxSELU(long double input);
    long double fxSwish(long double input);
}

#endif