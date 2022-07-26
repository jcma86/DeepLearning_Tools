#include "cmHelper.hpp"

using namespace cmHelper;

void Array::randomInit(size_t n, double *output, double min, double max)
{
    uniform_real_distribution<double> unif(min, max);
    random_device rd;
    mt19937 gen(rd());

    for (size_t i = 0; i < n; i += 1)
        output[i] = unif(gen);
};
