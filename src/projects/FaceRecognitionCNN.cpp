#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../libs/cnn/cmCNN.hpp"
#include "../libs/helper/cmHelper.hpp"
#include "../libs/nn/cmActivationFunction.hpp"
#include "../libs/nn/cmNN.hpp"
#include "../libs/pso/cmPSO.hpp"

using namespace std;
using namespace cv;
using namespace cmNN;
using namespace cmCNN;
using namespace cmPSO;

namespace po = boost::program_options;

typedef struct {
  unsigned int x1;
  unsigned int y1;
  unsigned int width;
  unsigned int height;

  unsigned int centerX;
  unsigned int centerY;
  unsigned int x2;
  unsigned int y2;

  unsigned int blur;
  unsigned int expression;
  unsigned int illumination;
  unsigned int occlusion;
  unsigned int pose;
  unsigned int invalid;

  double data[900];
} Face;

typedef struct {
  string path;
  vector<Face> faces;
} ExampleInfo;

double lastBest = -99999999999999.99999;

vector<ExampleInfo> examples;
Swarm pso;
size_t particleToSet = 0;
size_t validFaces = 0;
size_t invalidFaces = 0;

NeuralNetworkConfiguration nnConfig;
CNeuralNetworkConfiguration cnnConfig;
double trainNN(void* psoParams) {
  CNeuralNetwork cnn;
  NeuralNetwork nn;

  psoFitnessFxParams* params = (psoFitnessFxParams*)psoParams;
  double fitness = 0.0;

  cnn.createCNeuronNetwork(&cnnConfig);
  cnn.setInputs(NULL);
  cnn.setKernels(&params->position[0]);

  nn.createNeuronNetwork(nnConfig.nInputs, nnConfig.nLayers,
                         nnConfig.neuronsPerLayer);
  nn.setWeightsRange(params->min, params->max);
  nn.setWeights(&params->position[cnn.getNumOfParamsNeeded()]);
  nn.setInputs(NULL);
  nn.setLayerActivationFunction(nnConfig.nLayers - 1, "fxSigmoid");

  size_t correctValid = 0;
  size_t incorrectValid = 0;

  size_t correctInvalid = 0;
  size_t incorrectInvalid = 0;

  size_t totalExamples = examples.size();
  for (size_t e = 0; e < totalExamples; e += 1) {
    size_t totalFaces = examples[e].faces.size();
    for (size_t f = 0; f < totalFaces; f += 1) {
      Face* face = &examples[e].faces[f];
      cnn.setInputs(face->data, true);
      cnn.compute();
      nn.setInputs(cnn.getOuput(), true);
      nn.compute(Z_SCORE, nnConfig.softMax);
      double* outNN = nn.getOutput();
      size_t outSize = nn.getOutputSize();

      if (outNN[0] >= nnConfig.minThreshold) {
        correctValid += face->invalid == 0 ? 1 : 0;
        incorrectInvalid += face->invalid == 1 ? 1 : 0;
      } else {
        correctInvalid += face->invalid == 1 ? 1 : 0;
        incorrectValid += face->invalid == 0 ? 1 : 0;
      }
    }
  }

  double a = (double)correctValid / (double)validFaces;
  double b = (double)correctInvalid / (double)invalidFaces;

  double c = (double)incorrectValid / (double)validFaces;
  double d = (double)incorrectInvalid / (double)invalidFaces;

  fitness = ((0.5 * a) + (0.5 * b) - (0.5 * c) - (0.5 * d)) * 100.0;

  return fitness;
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  string nnconfigpath;
  string cnnconfigpath;

  bool setpso = false;
  bool loadNNConfig = false;
  bool loadCNNConfig = false;
  desc.add_options()("help", "Lists valid options.")(
      "loadNNConfig", po::value<string>(), "Opens NN configuration from file.")(
      "loadCNNConfig", po::value<string>(),
      "Opens CNN configuration from file.")("setParticlePosition",
                                            po::value<size_t>(), "")(
      "swarmSize", po::value<int>(), "")("psoMinVel", po::value<double>(), "")(
      "psoMaxVel", po::value<double>(), "")("psoMinPos", po::value<double>(),
                                            "")(
      "psoMaxPos", po::value<double>(), "")("psoThreads", po::value<int>(), "");

  double minPos = -10.0;
  double maxPos = -10.0;
  double minVel = -0.5;
  double maxVel = -0.5;
  int psoThreads = 15;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  int swarmSize = 8;
  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }
  if (vm.count("loadNNConfig")) {
    nnconfigpath = vm["loadNNConfig"].as<string>();
    loadNNConfig = true;
  }
  if (vm.count("loadCNNConfig")) {
    cnnconfigpath = vm["loadCNNConfig"].as<string>();
    loadCNNConfig = true;
  }
  if (vm.count("swarmSize")) {
    swarmSize = vm["swarmSize"].as<int>();
  }
  if (vm.count("setParticlePosition")) {
    setpso = true;
    particleToSet = vm["setParticlePosition"].as<size_t>();
  }
  if (vm.count("psoMinVel")) {
    minVel = vm["psoMinVel"].as<double>();
  }
  if (vm.count("psoMaxVel")) {
    maxVel = vm["psoMaxVel"].as<double>();
  }
  if (vm.count("psoMinPos")) {
    minPos = vm["psoMinPos"].as<double>();
  }
  if (vm.count("psoMaxPos")) {
    maxPos = vm["psoMaxPos"].as<double>();
  }
  if (vm.count("psoThreads")) {
    psoThreads = vm["psoThreads"].as<int>();
  }

  fstream file;
  file.open(
      "/Users/jose/Downloads/WIDER_DATASET/wider_face_split/"
      "wider_face_train_bbx_gt.txt",
      ios::in);

  if (!file.is_open())
    return -1;

  string line;
  size_t count = 0;
  while (getline(file, line)) {
    ExampleInfo anExample;
    anExample.path.assign(line);
    getline(file, line);
    int nFaces = atoi(line.c_str());
    if (nFaces == 0)
      getline(file, line);
    for (int e = 0; e < nFaces; e += 1) {
      Face aFace;
      getline(file, line);
      istringstream lineToParse(line);
      lineToParse >> aFace.x1 >> aFace.y1 >> aFace.width >> aFace.height;
      lineToParse >> aFace.blur >> aFace.expression >> aFace.illumination >>
          aFace.invalid >> aFace.occlusion >> aFace.pose;
      aFace.x2 = aFace.x1 + aFace.width;
      aFace.y2 = aFace.y1 + aFace.height;
      aFace.centerX = aFace.x1 + (aFace.width / 2);
      aFace.centerY = aFace.y1 + (aFace.height / 2);

      if (aFace.x1 == aFace.x2 || aFace.y1 == aFace.y2)
        continue;

      if (aFace.invalid == 0)
        validFaces += 1;
      if (aFace.invalid == 1)
        invalidFaces += 1;

      Mat img = imread(
          "/Users/jose/Downloads/WIDER_DATASET/training/" + anExample.path, 0);
      uint8_t* ptr;

      Rect crop(aFace.x1, aFace.y1, aFace.width, aFace.height);
      Mat faceImg = img(crop);
      resize(faceImg, faceImg, Size(30, 30));

      ptr = (uint8_t*)faceImg.data;

      for (size_t i = 0; i < 900; i += 1)
        aFace.data[i] = (double)ptr[i] / 255.0;

      anExample.faces.push_back(aFace);
    }
    examples.push_back(anExample);
    if (examples.size() > 25)
      break;
  }
  printf("Valid faces  : %ld\n", validFaces);
  printf("Invalid faces: %ld\n", invalidFaces);

  CNeuralNetwork::loadConfiguration(cnnconfigpath.c_str(), &cnnConfig);
  CNeuralNetwork cnntmp;
  cnntmp.createCNeuronNetwork(&cnnConfig);
  CNeuronDataSize cnnOutSize = cnntmp.getOutputSize();

  NeuralNetwork::loadConfiguration(nnconfigpath.c_str(), &nnConfig,
                                   cnnOutSize.d * cnnOutSize.w * cnnOutSize.h);

  cnnConfig.nParams = cnntmp.getNumOfParamsNeeded();
  printf("CNN Params needed: %ld\n", cnntmp.getNumOfParamsNeeded());
  printf("NN Weights needed: %ld\n", nnConfig.nWeights);
  printf("SoftMax: %s\n", nnConfig.softMax ? "TRUE" : "FALSE");

  /* PSO CREATION - START */
  size_t particleDim = cnntmp.getNumOfParamsNeeded() + nnConfig.nWeights;
  size_t swarmDim = pso.createSwarm(swarmSize, particleDim);
  double* position = new double[swarmDim];
  double* velocity = new double[swarmDim];

  printf("Creating vectors %ld\n", swarmDim);
  cmHelper::Array::randomInit(swarmDim, position, nnConfig.minW, nnConfig.maxW);
  cmHelper::Array::randomInit(swarmDim, velocity, minVel, maxVel);
  printf("Creating vectors complete\n");

  pso.setFitnessFunction(trainNN);
  pso.initPosition(position, nnConfig.minW, nnConfig.maxW);
  pso.initVelocity(velocity, minVel, maxVel);
  pso.initWeights(0.729, 1.4944, 1.4944);
  if (setpso) {
    printf("Setting particle position from NN file.\n");
    pso.setParticlePosition(particleToSet, nnConfig.weights);
  }
  pso.setPrecision(15);
  /* PSO CREATION - END */

  int shakeCountdown = 20;
  for (int g = 0; g < 1000; g += 1) {
    pso.compute(psoThreads);
    printf("\r%4d - Best fitness: %.15lf (p: %ld)\n", g, pso.getBestFitness(),
           pso.getBestParticle());
    if (lastBest != pso.getBestFitness()) {
      lastBest = pso.getBestFitness();
      stringstream stream;
      stream << fixed << setprecision(5) << lastBest;
      string nnpath = "bests/nn_" + stream.str() + ".txt";
      string cnnpath = "bests/cnn_" + stream.str() + ".txt";

      double* pos = pso.getBestPosition();
      CNeuralNetwork::saveToFile(cnnpath.c_str(), pos, &cnnConfig);
      NeuralNetwork::saveToFile(nnpath.c_str(),
                                &pos[cnntmp.getNumOfParamsNeeded()], &nnConfig);
      shakeCountdown = 20;
    } else
      shakeCountdown -= 1;

    if (shakeCountdown == 0) {
      pso.shakeSwarm();
      shakeCountdown = 20;
    } else
      pso.evolve();

    // char key = (char)waitKey(1);
    // if (key == 27)
    //     break;
    exit(0);
  }

  delete[] nnConfig.neuronsPerLayer;
  delete[] nnConfig.weights;
  delete[] position;
  delete[] velocity;

  return 0;
}