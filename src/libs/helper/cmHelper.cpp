#include "cmHelper.hpp"

using namespace cmHelper;

void Array::randomInit(size_t n, double *output, double min, double max, double precision)
{
    uniform_real_distribution<double> unif(min, max);
    random_device rd;
    mt19937 gen(rd());

    double mPre = pow(10, precision);

    for (size_t i = 0; i < n; i += 1)
        output[i] = floor(unif(gen) * mPre) / mPre;
};
