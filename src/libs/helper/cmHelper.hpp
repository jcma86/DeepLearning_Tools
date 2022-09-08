#ifndef __CM_LIBS_HELPER__
#define __CM_LIBS_HELPER__

#include <math.h>
#include <stdio.h>
#include <random>

using namespace std;

namespace cmHelper {
class Array {
 public:
  static void randomInit(size_t n,
                         double* output,
                         double min = -1.0,
                         double max = 1.0,
                         double precision = 15.0);
};
class Output {
 public:
  static void gotoxy(int x, int y);
};
}  // namespace cmHelper

#endif