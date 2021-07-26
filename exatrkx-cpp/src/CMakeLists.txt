cmake_minimum_required(VERSION 3.17)

project(ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

#cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

set(CMAKE_CUDA_ARCHITECTURES "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA  REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
#find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)
#find_package(cudf REQUIRED) 
#find_package(cugraph REQUIRED)

add_executable(inference inferencex.cpp)
#target_compile_features(inference PRIVATE cxx_std_17)
target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${CUDA_RUNTIME_LIB}  faiss xtensor xtensor::optimize xtensor::use_xsimd)