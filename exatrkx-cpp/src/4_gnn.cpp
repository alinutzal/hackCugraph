//module load gnu
//module load cmake
//module load cuda/11.0.3
//source activate xeus
// cd onnx_run
//cmake -B build
//cmake --build build --config Release --parallel
//./build/src/inference  --use_cuda
#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

using namespace xt::placeholders;  // required for `_` to work


#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
    {
        useCUDA = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
    // #include "cuda_provider_factory.h"
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    printf("Using Onnxruntime C++ API\n");
    const char* model_path = "datanmodels/e_model_full.onnx";
    Ort::Session session(env, model_path, session_options);
    
    
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                     // Otherwise need vector<vector<>>

    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> output_node_dims; 

    printf("Number of inputs = %zu\n", num_input_nodes);
    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
    output_node_names = {"output1"};
    

    //read input data
    std::ifstream f ("datanmodels/in_e1000.csv");   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(argv[1])).c_str());
        return 1;
    }
    std::string line;                    /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<float> input_tensor_values;      /* vector of vector<float> for 2d array */

    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        input_tensor_values.insert (input_tensor_values.end(),row.begin(),row.end());  
    }
    f.close();

    std::cout << "complete array\n\n";
    //for (auto& val : input_tensor_values)           /* iterate over vals */
    //    std::cout << val << "  ";        /* output value      */
    std::cout << "\n";                   /* tidy up with '\n' */

    size_t input_tensor_size = input_tensor_values.size();
    std::cout << input_tensor_size* sizeof(float) << "\n";
    input_node_dims[0] = input_tensor_size/12;
    std::cout << input_node_dims[0] << " " << input_node_dims[1] <<"\n";

    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 2);
    assert(input_tensor.IsTensor());

    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {floatarr, floatarr+input_tensor_size*8/12};

    std::vector<std::size_t> shape = { input_tensor_size/12, 8 };
    std::cout << shape[0] << " " <<shape[1]<<"\n";

    std::cout << "complete array\n\n";
    int i = 0;
    for (auto& val : output_tensor_values) {          /* iterate over vals */
        //std::cout << val << "  ";        /* output value      */
        i++;
        if (i % 8 == 0) {
            //std::cout << "\n";   
            }                /* tidy up with '\n' */
    }
    std::cout << i/8 << "\n";
    
    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, 1, output_node_names.data(),1);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                static_cast<float>(numTests) << " ms" << std::endl;
}