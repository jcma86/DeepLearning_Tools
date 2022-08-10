#ifndef __CM_LIBS_CNN__
#define __CM_LIBS_CNN__

#include <stdio.h>
#include <stdint.h>

namespace cmCNN
{
    class CNeuron
    {
    private:
        double *_input = NULL;
        char _activationFXName[50];

        size_t _inW = 1;
        size_t _inH = 1;
        size_t _inD = 1;

        size_t _kW = 1;
        size_t _kH = 1;
        size_t _kD = 1;
        size_t _l = 1;

        double *_kernel = NULL;
        uint8_t _stride = 1;
        uint8_t _maxPoolingSize = 1;

        double (*_activationFunction)(double) = NULL;

        double *_output = NULL;

        void releaseMemory();
        bool isReady();

    public:
        CNeuron() {}
        ~CNeuron();

        void setInput(double *input, size_t w = 1, size_t h = 1, size_t d = 1);
        void setKernel(double *kernel, size_t w = 1, size_t h = 1, size_t d = 1, size_t l = 1);
        void setStride(uint8_t stride);
        void setMaxPoolingSize(uint8_t maxPoolSize);
        void setActivationFunction(double (*_activationFunction)(double));
        void init();

        size_t getNumOfParamsNeeded();
        size_t getOutputWidth();
        size_t getOutputHeight();
        size_t getOutputDepth();
        double *getOutput();

        void compute();
    };
}

#endif
