#include <stdio.h>
#include <stdint.h>

int main()
{
    double _input[16] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double _kernel[9] = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};

    int dStart = 1 / 2;
    int wStart = 3 / 2;
    int hStart = 3 / 2;

    size_t _inD = 1;
    size_t _inW = 4;
    size_t _inH = 4;
    size_t _kD = 1;
    size_t _kW = 3;
    size_t _kH = 3;
    size_t _stride = 1;
    size_t outW = 2;
    size_t outC = 1;

    double *_output = new double[4];

    for (int indexD = dStart; indexD < (int)(_inD - dStart); indexD += _stride)
    {
        for (int indexH = hStart; indexH < (_inH - hStart); indexH += _stride)
        {
            for (int indexW = wStart; indexW < (_inW - wStart); indexW += _stride)
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

    int p = 0;
    printf("\n");
    for (int i = 0; i < 2; i += 1)
    {
        for (int j = 0; j < 2; j += 1)
        {
            printf("%+.3lf   ", _output[p]);
            p += 1;
        }
        printf("\n");
    }

    delete[] _output;

    return 0;
}