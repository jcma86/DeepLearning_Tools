#include <stdint.h>
#include <stdio.h>
#include <iostream>

#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../libs/cmDeepLearning.hpp"

using namespace std;
using namespace cmCNN;
namespace po = boost::program_options;

CNeuralNetworkConfiguration cnnConfig;
int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  string cnnconfigpath;
  bool setpso = false;
  bool loadNNConfig = false;
  desc.add_options()("help", "Lists valid options.")(
      "load-cnn-config", po::value<string>(),
      "Opens CNN configuration from file.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }
  if (vm.count("load-cnn-config")) {
    cnnconfigpath = vm["load-cnn-config"].as<string>();
    loadNNConfig = true;
  }

  CNeuralNetwork::loadConfiguration(cnnconfigpath.c_str(), &cnnConfig);
  CNeuralNetwork m;
  m.createCNeuronNetwork(&cnnConfig);

  // size_t inSize =
  //     cnnConfig.inputSize.d * cnnConfig.inputSize.w * cnnConfig.inputSize.h;
  // double* input = new double[inSize];

  // for (size_t i = 0; i < inSize; i += 1) {
  //   input[i] = 2.0 / (i + 1);
  // }

  // m.setInputs(input);
  // m.setKernels(cnnConfig.params);
  // m.compute();

  CNeuronDataSize s = m.getOutputSize();
  printf("Output size: %ld,%ld,%ld ---> %ld params needed.\n", s.d, s.w, s.h,
         m.getNumOfParamsNeeded());

  // size_t i = 0;
  // for (size_t d = 0; d < cnnConfig.inputSize.d; d += 1) {
  //   for (size_t h = 0; h < cnnConfig.inputSize.h; h += 1) {
  //     printf("\n");
  //     for (size_t w = 0; w < cnnConfig.inputSize.w; w += 1) {
  //       printf("   %2.3lf", input[i]);
  //       i += 1;
  //     }
  //   }
  // }

  // i = 0;
  // printf("\n");
  // printf("\n");
  // double* nno = m.getOuput();
  // for (size_t d = 0; d < s.d; d += 1) {
  //   for (size_t h = 0; h < s.h; h += 1) {
  //     printf("\n");
  //     for (size_t w = 0; w < s.w; w += 1) {
  //       printf("   %2.3lf", nno[i]);
  //       i += 1;
  //     }
  //   }
  // }

  printf("\n\n");

  // delete[] input;

  return 0;
}