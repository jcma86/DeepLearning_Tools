#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../libs/cnn/cmCNN.hpp"
#include "../libs/helper/cmHelper.hpp"
#include "../libs/nn/cmActivationFunction.hpp"
#include "../libs/nn/cmNN.hpp"

using namespace std;
using namespace cmNN;
using namespace cmCNN;
using namespace cv;

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

  double* data;
} Face;

typedef struct {
  string path;
  vector<Face> faces;
} ExampleInfo;

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
                                            po::value<size_t>(), "");

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

  NeuralNetworkConfiguration nnConfig;
  CNeuralNetworkConfiguration cnnConfig;
  CNeuralNetwork cnn;
  NeuralNetwork nn;

  vector<ExampleInfo> examples;
  size_t particleToSet = 0;
  size_t validFaces = 0;
  size_t invalidFaces = 0;

  CNeuralNetwork::loadConfiguration(cnnconfigpath.c_str(), &cnnConfig);
  cnn.createCNeuronNetwork(&cnnConfig);
  CNeuronDataSize cnnOutSize = cnn.getOutputSize();

  NeuralNetwork::loadConfiguration(nnconfigpath.c_str(), &nnConfig,
                                   cnnOutSize.d * cnnOutSize.w * cnnOutSize.h);

  fstream file;
  file.open(
      "/Users/jose/Downloads/WIDER_DATASET/wider_face_split/"
      "wider_face_train_bbx_gt.txt",
      ios::in);

  if (!file.is_open())
    return -1;

  string line;
  size_t count = 0;
  printf("Loading face examples.\n");
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
      resize(faceImg, faceImg,
             Size(cnnConfig.inputSize.w, cnnConfig.inputSize.h));

      ptr = (uint8_t*)faceImg.data;

      aFace.data = new double[cnnConfig.inputSize.w * cnnConfig.inputSize.h];
      for (size_t i = 0; i < (cnnConfig.inputSize.w * cnnConfig.inputSize.h);
           i += 1)
        aFace.data[i] = (double)ptr[i] / 255.0;

      anExample.faces.push_back(aFace);
    }
    examples.push_back(anExample);
    // if (examples.size() > 35)
    //   break;
  }
  printf("Valid faces  : %ld\n", validFaces);
  printf("Invalid faces: %ld\n", invalidFaces);

  cnnConfig.nParams = cnn.getNumOfParamsNeeded();
  printf("CNN Params needed: %ld\n", cnn.getNumOfParamsNeeded());
  printf("NN Weights needed: %ld\n", nnConfig.nWeights);
  printf("SoftMax: %s\n", nnConfig.softMax ? "TRUE" : "FALSE");

  cnn.setInputs(NULL);
  cnn.setKernels(cnnConfig.params);

  nn.createNeuronNetwork(nnConfig.nInputs, nnConfig.nLayers,
                         nnConfig.neuronsPerLayer);
  nn.setWeightsRange(nnConfig.minW, nnConfig.maxW);
  nn.setWeights(nnConfig.weights);
  nn.setInputs(NULL);
  nn.setLayerActivationFunction(nnConfig.nLayers - 1, "fxSigmoid");

  double fitness = 0.0;

  size_t correctValid = 0;
  size_t incorrectValid = 0;
  size_t correctInvalid = 0;
  size_t incorrectInvalid = 0;

  size_t totalFaces = validFaces + invalidFaces;

  printf("\e[1;1H\e[2J");
  printf(
      "   Fitness:       Valid Faces:       Invalid Faces:       Total "
      "Valid/Invalid (processed)\n");
  for (size_t e = 0; e < examples.size(); e += 1) {
    for (size_t f = 0; f < examples[e].faces.size(); f += 1) {
      Face* face = &examples[e].faces[f];
      cnn.setInputs(face->data, true);
      cnn.compute();
      nn.setInputs(cnn.getOuput());
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

      double a = (double)correctValid / (double)validFaces;
      double b = (double)correctInvalid / (double)invalidFaces;

      double c = (double)incorrectValid / (double)validFaces;
      double d = (double)incorrectInvalid / (double)invalidFaces;

      fitness = ((0.5 * a) + (0.5 * b) - (0.5 * c) - (0.5 * d)) * 100.0;

      cmHelper::Output::gotoxy(3, 2);
      printf("%3.5lf    ", fitness);
      cmHelper::Output::gotoxy(19, 2);
      printf("%5ld/%ld    ", correctValid, incorrectValid);
      cmHelper::Output::gotoxy(39, 2);
      printf("%5ld/%ld    ", correctInvalid, incorrectInvalid);
      cmHelper::Output::gotoxy(59, 2);
      printf(
          "%4ld/%ld --> %ld (%3.2lf %%)", validFaces, invalidFaces, totalFaces,
          (correctValid + incorrectValid + correctInvalid + incorrectInvalid) *
              100.0 / (double)totalFaces);
      cmHelper::Output::gotoxy(1, 3);
      printf("\n");
    }
  }

  delete[] nnConfig.neuronsPerLayer;
  delete[] nnConfig.weights;
  delete[] cnnConfig.params;
  delete[] cnnConfig.layerConfig;

  return 0;
}