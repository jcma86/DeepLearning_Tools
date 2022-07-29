cmake_minimum_required(VERSION 3.9)
project(cmDeepLearning VERSION 1.0.1 DESCRIPTION "DeepLearning tools")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_C_FLAGS "-Wall")

find_package(OpenCV REQUIRED)

# Libraries
add_library(cmHelper STATIC ./libs/helper/cmHelper.cpp)
add_library(cmActivationFunction STATIC ./libs/nn/cmActivationFunction.cpp)
add_library(cmNN STATIC ./libs/nn/cmNN.cpp)
add_library(cmPSO STATIC ./libs/pso/cmPSO.cpp)

target_link_libraries(${cmNN} cmActivationFunction)

# Execs Test
set(TEST_NN NeuralNetworkTest)
add_executable(${TEST_NN} tests/NeuralNetwork.cpp)
target_link_libraries(${TEST_NN} m cmNN cmActivationFunction)

set(TEST_ImageInput ImageInputTest)
add_executable(${TEST_ImageInput} tests/ImageInput.cpp)
target_link_libraries(${TEST_ImageInput} m ${OpenCV_LIBS})

set(TEST_WebCam WebCamTest)
add_executable(${TEST_WebCam} tests/WebCamInput.cpp)
target_link_libraries(${TEST_WebCam} m ${OpenCV_LIBS})

set(TEST_ImgNN ImgNNTest)
add_executable(${TEST_ImgNN} tests/ImgNN.cpp)
target_link_libraries(${TEST_ImgNN} m ${OpenCV_LIBS} cmHelper cmNN cmActivationFunction)

set(TEST_PSO SimplePSOTest)
add_executable(${TEST_PSO} tests/SimplePSO.cpp)
target_link_libraries(${TEST_PSO} m cmHelper cmPSO)

set(TEST_VideoInput VideoInputTest)
add_executable(${TEST_VideoInput} tests/VideoInput.cpp)
target_link_libraries(${TEST_VideoInput} m ${OpenCV_LIBS})

set(TEST_Convolution ConvolutionTest)
add_executable(${TEST_Convolution} tests/Convolution.cpp)
target_link_libraries(${TEST_Convolution} m)


set(PROJECT_FRAMEGEN FrameGenerator)
add_executable(${PROJECT_FRAMEGEN} projects/FrameGenerator.cpp)
target_link_libraries(${PROJECT_FRAMEGEN} m ${OpenCV_LIBS} cmHelper cmActivationFunction cmNN cmPSO)
