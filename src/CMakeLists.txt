cmake_minimum_required(VERSION 3.9)
project(cmDeepLearning VERSION 1.0.1 DESCRIPTION "DeepLearning tools")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "-Wall")

include(/Users/jose/vcpkg/scripts/buildsystems/vcpkg.cmake)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(websocketpp CONFIG REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# Libraries
add_library(cmHelper STATIC ./libs/helper/cmHelper.cpp)
add_library(cmActivationFunction STATIC ./libs/nn/cmActivationFunction.cpp)
add_library(cmNN STATIC ./libs/nn/cmNN.cpp)
add_library(cmCNN STATIC ./libs/cnn/cmCNN.cpp)
add_library(cmPSO STATIC ./libs/pso/cmPSO.cpp)

target_link_libraries(${cmNN} cmActivationFunction)
# target_link_libraries(${cmPSO} m pthread)

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
target_link_libraries(${TEST_Convolution} m cmCNN Boost::program_options)

set(TEST_SocketServer SocketServerTest)
add_executable(${TEST_SocketServer} tests/SocketServer.cpp)
target_link_libraries(${TEST_SocketServer} m websocketpp::websocketpp)

set(TEST_SocketClient SocketClientTest)
add_executable(${TEST_SocketClient} tests/SocketClient.cpp)
target_link_libraries(${TEST_SocketClient} m websocketpp::websocketpp)


set(PROJECT_FRAMEGEN FrameGenerator)
add_executable(${PROJECT_FRAMEGEN} projects/FrameGenerator.cpp)
target_link_libraries(${PROJECT_FRAMEGEN} m ${OpenCV_LIBS} cmHelper cmNN cmActivationFunction cmPSO)

set(PROJECT_FACERECOG FaceRecognition)
add_executable(${PROJECT_FACERECOG} projects/FaceRecognition.cpp)
target_link_libraries(${PROJECT_FACERECOG} m ${OpenCV_LIBS} cmHelper cmNN cmActivationFunction cmPSO Boost::program_options)

set(PROJECT_FACERECOG_CNN FaceRecognitionCNN)
add_executable(${PROJECT_FACERECOG_CNN} projects/FaceRecognitionCNN.cpp)
target_link_libraries(${PROJECT_FACERECOG_CNN} m ${OpenCV_LIBS} cmHelper cmNN cmCNN cmActivationFunction cmPSO Boost::program_options)

set(PROJECT_FACERECOG_CNN_TEST FaceRecognitionCNNTest)
add_executable(${PROJECT_FACERECOG_CNN_TEST} projects/FaceRecognitionCNNTest.cpp)
target_link_libraries(${PROJECT_FACERECOG_CNN_TEST} m ${OpenCV_LIBS} cmHelper cmNN cmCNN cmActivationFunction Boost::program_options)

