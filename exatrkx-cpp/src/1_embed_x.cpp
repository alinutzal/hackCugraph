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
#include <algorithm>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

//#include <torch/torch.h>

using namespace xt::placeholders;  // required for `_` to work

#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "cuda.h"
//#include <cuda_fp16.h>
#include "cuda_runtime_api.h"

struct EmbeddingInferParams
{
    int batchSize;              //!< The input height
    int inputW;              //!< The input width
    int outputSize;          //!< The output size
    std::string modelFile; //!< The filename of the weights file
};

class EmbeddingInfer
{
public:
    EmbeddingInfer(const EmbeddingInferParams& params){}
    EmbeddingInfer(){}
    Ort::Session initOnnxSession(Ort::Env& env);
    std::vector<float> infer(Ort::Session& session);
    std::vector<const char*> getInputNodes(Ort::Session& session);
    std::vector<const char*> getOutputNodes(Ort::Session& session);
    bool teardown();
    bool verifyOutput(std::vector<float>);
private:
    std::vector<float> processInput();
};

Ort::Session EmbeddingInfer::initOnnxSession(Ort::Env& env){
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
    return session;
}

 std::vector<const char*> EmbeddingInfer::getInputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                     // Otherwise need vector<vector<>>
     
    printf("Number of inputs = %zu\n", num_input_nodes);

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
    return input_node_names;    
}

 std::vector<const char*> EmbeddingInfer::getOutputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> output_node_dims; 

    printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Input %d : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Output %d : type=%d\n", i, type);

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    for (int j = 0; j < output_node_dims.size(); j++)
      printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    return output_node_names;    
}

std::vector<float> EmbeddingInfer::infer(Ort::Session& session){
    //std::cout<<"got before";
    std::vector<float> input_tensor_values = processInput();

    size_t input_tensor_size = input_tensor_values.size();
   
    //for (int i=0; i<input_tensor_size; i++)
    //   std::cout << input_tensor_values[i] << " ";
    //std::cout<<"got after";
     
    //size_t input_tensor_size = input_tensor_values.size();
    std::vector<int64_t> input_node_dims(2); 
    input_node_dims[1] = 12;
    input_node_dims[0] = input_tensor_size/input_node_dims[1];
   
    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    //std::vector<const char*> input_node_names(num_input_nodes);
    //input_node_names = {"actual_input_1"};        

    size_t num_output_nodes = session.GetOutputCount();
    //std::vector<const char*> output_node_names(num_output_nodes);
    //output_node_names = {"output1"};    
    //std::vector<int64_t> output_node_dims; 
    
    // create input tensor object from data values
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, 
    //                                                          input_node_dims.data(), 2);
    //Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, values_x.data(), input_tensor_size, 
    //                                                          input_node_dims.data(), 2);//, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    
    Ort::MemoryInfo memory_info_cuda("Cuda",OrtArenaAllocator,0, OrtMemTypeDefault);
    Ort::Allocator memory_allocator(session, memory_info_cuda);
    void* input_data = memory_allocator.Alloc(sizeof(float) * input_tensor_size);
    cudaMemcpy(input_data, input_tensor_values.data(), sizeof(float) * input_tensor_size, cudaMemcpyHostToDevice);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_allocator.GetInfo(),reinterpret_cast<float*>(input_data), input_tensor_size, 
                                                              input_node_dims.data(), 2);

    assert(input_tensor.IsTensor());

    std::vector<const char*> input_node_names = getInputNodes(session);
    std::vector<const char*> output_node_names = getOutputNodes(session);
    // score model & input tensor, get back output tensor
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {floatarr, floatarr+ input_node_dims[0]*8};
    
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
    return output_tensor_values;
}

bool EmbeddingInfer::teardown(){
    return true;
}
std::vector<float> EmbeddingInfer::processInput(){
    std::ifstream in_file;
    in_file.open("datanmodels/in_e1000.csv");
    auto input = xt::load_csv<double>(in_file);
    std::cout << input <<"\n";
    auto shape = input.shape();   // shape = {2, 3}
    std::cout << shape[0] << " " << shape[1] << "\n";
    
    std::vector<float> input_tensor_values(input.begin(), input.end());
    size_t input_tensor_size = input_tensor_values.size();
    std::cout << input_tensor_size << "\n";
    //for (int i=0; i<input_tensor_size; i++)
    //   std::cout << input_tensor_values[i] << " ";
    //input_node_dims[0] = input_tensor_size/12;
    //std::cout << input_node_dims[0] << " " << input_node_dims[1] <<"\n";
    return input_tensor_values;
}

bool EmbeddingInfer::verifyOutput(std::vector<float> output_tensor_values){
    size_t tensor_size = output_tensor_values.size();
    long unsigned int dim0 = tensor_size/8;
    std::vector<std::size_t> shape = {dim0, 8};
    auto output = xt::adapt(output_tensor_values, shape);
    std::cout << output <<"\n";
    std::cout << dim0 <<" "<< 8 <<"\n";
    
    std::ifstream in_file;
    in_file.open("datanmodels/out_e1000.csv");
    auto input = xt::load_csv<double>(in_file);
    std::cout << input <<"\n";
    auto shapeo = input.shape();   // shape = {2, 3}
    std::cout << shapeo[0] << " " << shapeo[1] << "\n";
    bool res = (output == input);
    return true;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./1_embed [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mlp/, data/mlp/)"
              << std::endl;
    std::cout << "--use_cuda          Run on the GPU." << std::endl;
    std::cout << "--use_cpu          Run on the CPU." << std::endl;    
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    if (argc <=1) {
        std::cout << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;        
    }
    else {
        if(cmdOptionExists(argv, argv+argc, "-h"))
        {
            printHelpInfo();
            return EXIT_SUCCESS;
        }
    }
    std::cout << "Building and running a GPU inference engine for Embedding" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    EmbeddingInfer embed; //initializeSampleParams(args));
    
    Ort::Session session = embed.initOnnxSession(env);
    std::cout << "here"<<"\n";
    //torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << "Tensor: "<< tensor << std::endl;
    
    std::vector<float> output_tensor_values = embed.infer(session);
    //size_t output_tensor_size = output_tensor_values.size();
    //for (int i=0; i<output_tensor_size; i++)
    //   std::cout << output_tensor_values[i] << " ";
    bool res = embed.verifyOutput(output_tensor_values);
    std::cout << "Output correct: " << res << "\n";
    return 0;
}

