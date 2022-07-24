cmake_minimum_required(VERSION 3.9)
project(cmDeepLearning VERSION 1.0.1 DESCRIPTION "DeepLearning tools")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_C_FLAGS "-Wall")

find_package(OpenCV REQUIRED)

# Libraries
add_library(cmNeuralNetwork STATIC ./libs/neural_networks/cmNeuralNetwork.cpp)
add_library(cmActivationFunction STATIC ./libs/neural_networks/cmActivationFunction.cpp)

# Execs Test
set(TEST_NN NeuralNetworkTest)
add_executable(${TEST_NN} tests/NeuralNetwork.cpp)
target_link_libraries(${TEST_NN} m cmNeuralNetwork cmActivationFunction)

set(TEST_ImageInput ImageInputTest)
add_executable(${TEST_ImageInput} tests/ImageInput.cpp)
target_link_libraries(${TEST_ImageInput} m ${OpenCV_LIBS})

set(TEST_WebCam WebCamTest)
add_executable(${TEST_WebCam} tests/WebCamInput.cpp)
target_link_libraries(${TEST_WebCam} m ${OpenCV_LIBS})

set(TEST_ImgNN ImgNNTest)
add_executable(${TEST_ImgNN} tests/ImgNN.cpp)
target_link_libraries(${TEST_ImgNN} m ${OpenCV_LIBS} cmNeuralNetwork cmActivationFunction)
