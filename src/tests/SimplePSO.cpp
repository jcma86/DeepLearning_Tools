#include <stdio.h>
#include "../libs/helper/cmHelper.hpp"
#include "../libs/pso/cmPSO.hpp"

using namespace cmPSO;

double fitness(void *params)
{
    psoFitnessFxParams *p = (psoFitnessFxParams *)params;
    double fitness = 0.0;

    for (size_t i = 0; i < p->nDimensions; i += 1)
        fitness += p->position[i] * p->position[i];

    return fitness;
}

int main()
{
    Swarm aSwarm;

    double *position;
    double *velocity;

    size_t population = 40;
    size_t dim = 2;

    size_t sdim = aSwarm.createSwarm(population, dim, false);
    printf("Total dimesion: %ld\n", sdim);

    position = new double[sdim];
    velocity = new double[sdim];

    cmHelper::Array::randomInit(sdim, position, -100.0, 100.0);
    cmHelper::Array::randomInit(sdim, velocity, -0.5, 0.5);

    aSwarm.setFitnessFunction(fitness);
    aSwarm.initPosition(position, -100.0, 100.0);
    aSwarm.initVelocity(velocity, -1.5, 1.5);
    aSwarm.initWeights(0.729, 1.4944, 1.4944);

    for (int g = 0; g < 500; g += 1)
    {
        aSwarm.compute();
        printf("\nBest fitness: %.15lf: ", aSwarm.getBestFitness());
        double *bp = aSwarm.getBestPosition();
        for (size_t d = 0; d < dim; d += 1)
            printf("%+.15lf    ", bp[d]);
        aSwarm.evolve();
    }

    delete[] position;
    delete[] velocity;

    return 0;
}