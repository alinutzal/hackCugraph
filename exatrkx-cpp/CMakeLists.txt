cmake_minimum_required(VERSION 3.20)

project(ExatrkX_ONNX_Runtime VERSION 0.0.1 LANGUAGES CXX)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Pass options to NVCC
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_70,code=sm_70; --std=c++17)

#set(CMAKE_CUDA_ARCHITECTURES 70)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_PREFIX_PATH "/users/PLS0129/ysu0053/libtorch_cxx11/share/cmake/Torch")

#find_package(CUDA REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
#find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
find_package(cudf REQUIRED) 
find_package(cugraph REQUIRED)
#find_package(rmm REQUIRED)

#SET(GCC_COVERAGE_LINK_FLAGS    "-lrt")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS "/users/PLS0129/ysu0053/.local/include/onnxruntime/")
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS "/users/PLS0129/ysu0053/.local/include/onnxruntime/")
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS "/users/PLS0129/ysu0053/.local/lib/onnxruntime/gpu")

#find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS "/global/homes/a/alazar/.local/include/onnxruntime/include/")
#find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS "/global/homes/a/alazar/.local/include/onnxruntime/include")
#find_library(ONNX_RUNTIME_LIB onnxruntime HINTS "/global/homes/a/alazar/.local/lib/onnxruntime/gpu")

#add_subdirectory(src)

#add_executable(1_embed EXCLUDE_FROM_ALL src/1_embed.cpp)
#target_include_directories(1_embed PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS})
#target_link_libraries(1_embed PRIVATE ${ONNX_RUNTIME_LIB} ${TORCH_LIBRARIES} cudart)


#add_executable(2_build_edge EXCLUDE_FROM_ALL src/2_build_edge.cpp)
#target_compile_features(2_build_edge PRIVATE cxx_std_17)
#target_include_directories(2_build_edge PRIVATE ${CUDA_INCLUDE_DIRS})
#target_link_libraries(2_build_edge PRIVATE ${CUDA_RUNTIME_LIB} ${TORCH_LIBRARIES} xtensor xtensor::optimize xtensor::use_xsimd)

#add_executable(2_5_build_edge EXCLUDE_FROM_ALL src/2_5_build_edge.cpp)
#target_compile_features(2_5_build_edge PRIVATE cxx_std_14)
#target_link_libraries(2_5_build_edge ${TORCH_LIBRARIES})
#set_property(TARGET 2_5_build_edge PROPERTY CXX_STANDARD 14)

#add_executable(example-app src/example-app.cpp)
#target_link_libraries(example-app ${TORCH_LIBRARIES})
#set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

#add_executable(3_filtering EXCLUDE_FROM_ALL src/3_filtering.cpp)
#target_compile_features(3_filtering PRIVATE cxx_std_17)
#target_include_directories(3_filtering PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(3_filtering PRIVATE ${ONNX_RUNTIME_LIB} ${TORCH_LIBRARIES} xtensor xtensor::optimize xtensor::use_xsimd cudart)

#add_executable(4_gnn EXCLUDE_FROM_ALL src/4_gnn.cpp)
#target_compile_features(4_gnn PRIVATE cxx_std_17)
#target_include_directories(4_gnn PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(4_gnn PRIVATE ${ONNX_RUNTIME_LIB} ${CUDA_RUNTIME_LIB} xtensor xtensor::optimize xtensor::use_xsimd)

#set(CUDF_INCLUDE_DIRS "/users/PLS0129/ysu0053/.conda/pkgs/libcudf-21.08.00a210707-cuda11.0_gf0f2170972_251/include/")
#set(CUGRAPH_INCLUDE_DIRS "/users/PLS0129/ysu0053/.conda/pkgs/libcugraph-0.19.0-cuda10.2_gd72b90b0_0/include/")
add_executable(5_labeling EXCLUDE_FROM_ALL src/5_labeling.cpp)
target_compile_features(5_labeling PRIVATE cxx_std_17)
target_include_directories(5_labeling PRIVATE ${CUDA_INCLUDE_DIRS} ${CUDF_INCLUDE_DIRS} ${CUGRAPH_INCLUDE_DIRS}) # ${RMM_INCLUDE_DIRS} ${CUGRAPH_INCLUDE_DIRS} ${SPL_INCLUDE_DIRS} #${CUGRAPH_INCLUDE_DIR2} )
target_link_libraries(5_labeling PRIVATE cudf cugraph)

#add_executable(6_test EXCLUDE_FROM_ALL src/6_test.cu)
#target_compile_features(6_test PRIVATE cxx_std_17)
#target_include_directories(6_test PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(6_test PRIVATE ${ONNX_RUNTIME_LIB} xtensor xtensor::optimize xtensor::use_xsimd)

#add_executable(inference EXCLUDE_FROM_ALL src/inference.cpp)
#target_compile_features(inference PRIVATE cxx_std_17)
#target_include_directories(inference PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(inference PRIVATE ${ONNX_RUNTIME_LIB} ${CUDA_RUNTIME_LIB})

#add_executable(inferencex EXCLUDE_FROM_ALL src/inferencex.cpp)
#target_compile_features(inferencex PRIVATE cxx_std_17)
#target_include_directories(inferencex PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(inferencex PRIVATE ${ONNX_RUNTIME_LIB} ${CUDA_RUNTIME_LIB} xtensor xtensor::optimize xtensor::use_xsimd)
