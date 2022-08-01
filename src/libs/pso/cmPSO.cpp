#include "cmPSO.hpp"
#include <random>
#include <pthread.h>

using namespace cmPSO;

Particle::~Particle()
{
    if (_bestPosition != NULL)
        delete[] _bestPosition;
}

bool Particle::hasDimension()
{
    if (_dimension <= 0)
        printf("[WARNING]: Number of dimension should be set first.");
    return _dimension > 0;
}

bool Particle::isReady()
{
    hasDimension();
    if (_position == NULL)
        printf("[ERROR]: No position vector has been set.");

    if (_velocity == NULL)
        printf("[ERROR]: No velocity vector has been set.");

    return _position != NULL && _velocity != NULL && hasDimension();
}

void Particle::evolve()
{
    if (!isReady())
        return;

    uniform_real_distribution<double> unif(0.0, 1.0);
    random_device rd;
    mt19937 gen(rd());

    for (size_t d = 0; d < _dimension; d += 1)
    {
        _position[d] += _velocity[d];
        _velocity[d] = (_iW * _velocity[d]) + ((_cW * unif(gen)) * (_bestPosition[d]) - _position[d]) / +((_sW * unif(gen)) * (_bestGlobalPosition[d]) - _position[d]);

        _velocity[d] = _position[d] > _maxP || _position[d] < _minP ? _velocity[d] * -1.0 : _velocity[d];
        _velocity[d] = _velocity[d] > _maxV ? _maxV : _velocity[d];
        _velocity[d] = _velocity[d] < _minV ? _minV : _velocity[d];

        _position[d] = _position[d] < _minP ? _minP : _position[d];
        _position[d] = _position[d] > _maxP ? _maxP : _position[d];
    }
}

void Particle::updateBest()
{
    if (!isReady()){
        printf("Particle not ready %ld\n", _id);
        return;
    }

    bool updateMax = false;
    bool updateMin = false;

    updateMax = _maximize && _fitness > _bestFitness;
    updateMin = !_maximize && _fitness < _bestFitness;

    if (updateMax || updateMin)
    {
        setBestFitness(_fitness);
        setBestPosition(_position);
    }
}

void Particle::setBestPosition(double *bestPosition, bool reallocate)
{
    if (hasDimension())
    {
        if (reallocate && _bestPosition)
        {
            delete[] _bestPosition;
            _bestPosition = NULL;
        }

        if (!_bestPosition)
            _bestPosition = new double[_dimension];

        for (size_t i = 0; i < _dimension; i += 1)
            _bestPosition[i] = bestPosition[i];
    }
}

void Particle::setBestGlobalPosition(double *position)
{
    if (hasDimension())
        _bestGlobalPosition = position;
}

void Particle::setBestFitness(double bestFitness)
{
    _bestFitness = bestFitness;
}

void Particle::setID(size_t id)
{
    _id = id;
}

void Particle::setDimension(size_t dimension)
{
    _dimension = dimension;
}

void Particle::setMaximize(bool maximize)
{
    _maximize = maximize;
    _bestFitness = maximize ? -99999999999.9999999 : 99999999999.9999999;
}

void Particle::setFitness(double fitness)
{
    _fitness = fitness;
}

void Particle::setFitnessFunction(double (*fitnessFunction)(void *))
{
    _fitnessFunction = fitnessFunction;
}

void Particle::compute()
{
    psoFitnessFxParams params;
    params.nDimensions = _dimension;
    params.position = _position;
    params.particleID = _id;

    if (isReady())
    {
        _fitness = _fitnessFunction((void *)&params);
        updateBest();
    }
}

void Particle::initWeights(double iW, double cW, double sW)
{
    _iW = iW;
    _cW = cW;
    _sW = sW;
}

void Particle::initPosition(double *position, double minP, double maxP)
{
    if (hasDimension())
    {
        _minP = minP < maxP ? minP : maxP;
        _maxP = maxP > minP ? maxP : minP;
        _position = position;
    }
}

void Particle::initVelocity(double *velocity, double minV, double maxV)
{
    if (hasDimension())
    {
        _minV = minV < maxV ? minV : maxV;
        _maxV = maxV > minV ? maxV : minV;
        _velocity = velocity;
    }
}

double *Particle::getPosition()
{
    return _position;
}

double *Particle::getVelocity()
{
    return _velocity;
}

double *Particle::getBestPosition()
{
    return _bestPosition;
}

double Particle::getBestFitness()
{
    return _bestFitness;
}

double Particle::getFitness()
{
    return _fitness;
}

// Swarm
Swarm::~Swarm()
{
    if (_particle)
        delete[] _particle;
    if (_bestPosition)
        delete[] _bestPosition;
    _particle = NULL;
    _bestPosition = NULL;
}

size_t Swarm::getSwarmDimension()
{
    return _dimension * _population;
}

void Swarm::setID(size_t id)
{
    _id = id;
}

void Swarm::setFitnessFunction(double (*fitnessFunction)(void *))
{
    for (size_t p = 0; p < _population; p += 1)
        _particle[p].setFitnessFunction(fitnessFunction);
}

size_t Swarm::createSwarm(size_t population, size_t dimension, bool maximize)
{
    if (_particle)
        delete[] _particle;
    if (_bestPosition)
        delete[] _bestPosition;

    _population = population;
    _dimension = dimension;
    _maximize = maximize;
    _particle = new Particle[_population];
    _bestPosition = new double[_dimension];

    for (size_t p = 0; p < _population; p += 1)
    {
        _particle[p].setID(p);
        _particle[p].setDimension(_dimension);
        _particle[p].setMaximize(_maximize);

        _particle[p].setBestGlobalPosition(_bestPosition);
    }
    _bestFitness = _particle[0].getBestFitness();

    return getSwarmDimension();
}

void Swarm::initPosition(double *position, double minP, double maxP)
{
    _position = position;
    for (size_t p = 0; p < _population; p += 1)
        _particle[p].initPosition(&_position[p * _dimension], minP, maxP);
}

void Swarm::initVelocity(double *velocity, double minV, double maxV)
{
    _velocity = velocity;
    for (size_t p = 0; p < _population; p += 1)
        _particle[p].initVelocity(&_velocity[p * _dimension], minV, maxV);
}

void Swarm::initWeights(double iW, double cW, double sW)
{
    for (size_t p = 0; p < _population; p += 1)
        _particle[p].initWeights(iW, cW, sW);
}

double *Swarm::getBestPosition()
{
    return _bestPosition;
}

double Swarm::getBestFitness()
{
    return _bestFitness;
}

size_t Swarm::getBestParticle()
{
    return _bestParticle;
}

void Swarm::updateBest()
{
    bool update = false;
    bool max = false;
    bool min = false;
    double tmpBF = _bestFitness;
    for (size_t p = 0; p < _population; p += 1)
    {
        max = _maximize && _particle[p].getBestFitness() > tmpBF;
        min = !_maximize && _particle[p].getBestFitness() < tmpBF;
        if (max || min)
        {
            tmpBF = _particle[p].getBestFitness();
            _bestParticle = p;
            update = true;
        }
    }
    if (update)
    {
        setBestFitness(_particle[_bestParticle].getBestFitness());
        setBestPosition(_particle[_bestParticle].getBestPosition());
    }
}

void Swarm::setBestPosition(double *bestPosition, bool reallocate)
{
    if (_bestPosition && reallocate)
        delete[] _bestPosition;
    if (!_bestPosition)
        _bestPosition = new double[_dimension];

    for (size_t d = 0; d < _dimension; d += 1)
        _bestPosition[d] = bestPosition[d];

    if (reallocate)
    {
        for (size_t p = 0; p < _population; p += 1)
            _particle[p].setBestGlobalPosition(_bestPosition);
    }
}

void Swarm::setBestFitness(double bestFitness)
{
    _bestFitness = bestFitness;
}

void *Swarm::evaluateThread(void *args)
{
    psoThreadArgs *params = (psoThreadArgs *)args;
    params->particle->compute();

    return NULL;
}

void Swarm::compute(size_t maxThreads)
{
    size_t thrPerBatch = maxThreads > _population ? _population : maxThreads;
    size_t thrCreated = 0;

    while (thrCreated < _population)
    {
        pthread_attr_t attr;
        pthread_t *threads = new pthread_t[thrPerBatch];
        psoThreadArgs *args = new psoThreadArgs[thrPerBatch];

        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        size_t created = 0;

        for (size_t ct = 0; ct < thrPerBatch; ct += 1)
        {
            args[created].particleID = thrCreated;
            args[created].particle = &_particle[thrCreated];
            int rc = pthread_create(&threads[created], &attr, &Swarm::evaluateThread, (void *)&args[created]);
            if (rc == 0)
            {
                thrCreated += 1;
                created += 1;
            }
        }

        printf("\rEvaluating %ld of %ld particles.", thrCreated, _population);
        fflush(stdout);

        for (size_t i = 0; i < created; i += 1)
            int rc = pthread_join(threads[i], NULL);

        thrPerBatch = thrCreated + thrPerBatch > _population ? _population - thrCreated : thrPerBatch;
        delete[] threads;
        delete[] args;
    }

    // for (size_t p = 0; p < _population; p += 1)
    // {
    //     printf("\rComputing Particle %ld/%ld", p + 1, _population);
    //     fflush(stdout);
    //     _particle[p].compute();
    // }
    updateBest();
}

void Swarm::evolve()
{
    for (size_t p = 0; p < _population; p += 1)
        _particle[p].evolve();
}
