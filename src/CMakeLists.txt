cmake_minimum_required(VERSION 3.9)
project(cmDeepLearning VERSION 1.0.1 DESCRIPTION "DeepLearning tools")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Libraries
add_library(cmNeuralNetwork STATIC ./libs/neural_networks/cmNeuralNetwork.cpp)
add_library(cmActivationFunction STATIC ./libs/neural_networks/cmActivationFunction.cpp)

# Execs Test
set(TEST_NN NeuralNetworkTest)
add_executable(${TEST_NN} tests/NeuralNetwork.cpp)
target_link_libraries(${TEST_NN} m cmNeuralNetwork cmActivationFunction)
