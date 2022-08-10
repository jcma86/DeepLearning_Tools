#include "cmCNN.hpp"

using namespace cmCNN;

CNeuron::~CNeuron()
{
    releaseMemory();
}

void CNeuron::releaseMemory()
{
    if (_output)
        delete[] _output;

    _output = NULL;
}

bool CNeuron::isReady()
{
    return _input && _kernel;
}

void CNeuron::setInput(double *input, size_t w, size_t h, size_t d)
{
    _inW = w;
    _inH = h;
    _inD = d;
    _input = input;
}

void CNeuron::setKernel(double *kernel, size_t w, size_t h, size_t d, size_t l)
{
    _kW = w;
    _kH = h;
    _kD = d;
    _kernel = kernel;
}

void CNeuron::setStride(uint8_t stride)
{
    _stride = stride;
}

void CNeuron::setActivationFunction(double (*activationFunction)(double))
{
    _activationFunction = activationFunction;
}

void CNeuron::init()
{
    if (!isReady())
    {
        printf("[ERROR]: You must set firts the input and kernel.\n");
        return;
    }

    releaseMemory();
    _output = new double[getOutputDepth() * getOutputHeight() * getOutputWidth()];
}

double *CNeuron::getOutput()
{
    return _output;
}

size_t CNeuron::getNumOfParamsNeeded()
{
    return _kW * _kH * _kD;
}

size_t CNeuron::getOutputWidth()
{
    return ((_inW - (_kW + ((_kW + 1) * (_l - 1)))) / _stride) + 1;
}

size_t CNeuron::getOutputHeight()
{
    return ((_inH - (_kH + ((_kH + 1) * (_l - 1)))) / _stride) + 1;
}

size_t CNeuron::getOutputDepth()
{
    return ((_inD - (_kD + ((_kD + 1) * (_l - 1)))) / _stride) + 1;
}

void CNeuron::compute()
{
    if (!isReady())
        return;

    size_t dStart = _kD / 2;
    size_t wStart = _kW / 2;
    size_t hStart = _kH / 2;

    size_t outW = getOutputWidth();
    size_t outC = getOutputDepth();

    size_t step = _stride + (_l - 1);

    // TODO: Add support for dilatation.
    for (int indexD = dStart + (_l - 1); indexD < (int)(_inD - dStart - (_l - 1)); indexD += step)
    {
        for (int indexH = hStart + (_l - 1); indexH < (int)(_inH - hStart - (_l - 1)); indexH += step)
        {
            for (int indexW = wStart + (_l - 1); indexW < (int)(_inW - wStart - (_l - 1)); indexW += step)
            {
                int outIndex = outC * ((outW * (indexH - hStart)) + (indexW - wStart)) + (indexD - dStart);
                _output[outIndex] = 0.0;

                for (int indexKD = -dStart; indexKD < (int)(_kD - dStart); indexKD += _stride)
                {
                    for (int indexKH = -hStart; indexKH < (int)(_kH - hStart); indexKH += _stride)
                    {
                        for (int indexKW = -wStart; indexKW < (int)(_kW - wStart); indexKW += _stride)
                        {
                            int indIn = _inD * ((_inW * (indexH + indexKH)) + (indexW + indexKW)) + (indexD + indexKD);
                            int indKe = _kD * ((_kW * (indexKH + hStart)) + (indexKW + wStart)) + (indexKD + dStart);

                            _output[outIndex] += _kernel[indKe] * _input[indIn];
                        }
                    }
                }
            }
        }
    }
}
