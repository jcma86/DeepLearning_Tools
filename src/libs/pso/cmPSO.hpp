#ifndef __CM_LIBS_PSO__
#define __CM_LIBS_PSO__

#include <stdint.h>
#include <stdio.h>

using namespace std;

namespace cmPSO {
class Particle {
 private:
  double _precision = 15.0;
  double _precisionMult = 1000000000000000.0;
  size_t _id;
  size_t _dimension = 0;
  bool _maximize = true;

  double _iW;
  double _cW;
  double _sW;

  double _minP = 1.0;
  double _maxP = -1.0;
  double _minV = -1.0;
  double _maxV = -1.0;

  double* _position = NULL;
  double* _velocity = NULL;
  double* _bestPosition = NULL;
  double* _bestGlobalPosition = NULL;
  double _fitness;
  double _bestFitness = -9999999999.99999;

  double (*_fitnessFunction)(void*) = NULL;

  bool hasDimension();
  bool isReady();

  void updateBest();

  void setBestPosition(double* bestPosition, bool reallocate = false);
  void setBestFitness(double bestFitness);

 public:
  Particle(){};
  ~Particle();

  void setID(size_t id);
  void setDimension(size_t dimension);
  void setMaximize(bool maximize = true);

  void setFitness(double fitness);
  void setFitnessFunction(double (*fitnessFunction)(void*) = NULL);
  void setBestGlobalPosition(double* position);
  void setPosition(double* position);
  void setPrecision(double precision = 15.0);
  void shakeParticle();

  void initWeights(double iW, double cW, double sW);
  void initPosition(double* position = NULL,
                    double minP = -1.0,
                    double maxP = 1.0);
  void initVelocity(double* velocity = NULL,
                    double minV = -0.05,
                    double maxV = 0.05);

  double* getPosition();
  double* getVelocity();
  double* getBestPosition();
  double getBestFitness();
  double getFitness();

  void compute();
  void evolve();
};

class Swarm {
 private:
  double _precision = 15.0;
  size_t _id;
  size_t _population;
  size_t _dimension = 0;
  size_t _bestParticle = 0;
  bool _maximize = true;

  Particle* _particle = NULL;

  double* _position = NULL;
  double* _velocity = NULL;
  double* _bestPosition = NULL;
  double _bestFitness = -9999999999.99999;

  void updateBest();
  void setBestPosition(double* bestPosition, bool reallocate = false);
  void setBestFitness(double bestFitness);

  static void* evaluateThread(void* args);

 public:
  Swarm(){};
  ~Swarm();

  size_t getSwarmDimension();

  void setID(size_t id);

  void setFitnessFunction(double (*fitnessFunction)(void*) = NULL);

  size_t createSwarm(size_t population, size_t dimension, bool maximize = true);
  void initWeights(double iW, double cW, double sW);
  void initPosition(double* position = NULL,
                    double minP = -1.0,
                    double maxP = 1.0);
  void initVelocity(double* velocity = NULL,
                    double minV = -0.05,
                    double maxV = 0.05);
  void setParticlePosition(size_t particle, double* position);
  void setPrecision(double precision = 15.0);
  void shakeSwarm();

  double* getBestPosition();
  double getBestFitness();
  double* getParticlePosition(size_t paricle);
  size_t getBestParticle();

  void compute(size_t maxThreads = 1);
  void evolve();
};

typedef struct {
  size_t particleID;
  size_t nDimensions;
  double* position;
  double min;
  double max;
} psoFitnessFxParams;

typedef struct {
  size_t particleID;
  Particle* particle;
} psoThreadArgs;
}  // namespace cmPSO

#endif