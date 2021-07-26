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

#include <torch/torch.h>
#include <torch/script.h>
using namespace torch::indexing;

#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "cuda.h"
#include <cuda_fp16.h>
#include "cuda_runtime_api.h"

struct FilteringInferParams
{
    int batchSize;              //!< The input height
    int inputW;              //!< The input width
    int outputSize;          //!< The output size
    std::string modelFile; //!< The filename of the weights file
};

class FilteringInfer
{
public:
    FilteringInfer(const FilteringInferParams& params){}
    FilteringInfer(){}
    Ort::Session initOnnxSession(Ort::Env& env);
    std::vector<float> infer(Ort::Session& session,std::vector<float>);
    std::vector<const char*> getInputNodes(Ort::Session& session);
    std::vector<const char*> getOutputNodes(Ort::Session& session);
    bool teardown();
    bool verifyOutput(std::vector<float>);
    std::vector<float>  processInput(long unsigned int& e_size);
};

Ort::Session FilteringInfer::initOnnxSession(Ort::Env& env){
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
    const char* model_path = "datanmodels/f_model_full.onnx";
    Ort::Session session(env, model_path, session_options);
    return session;
}

 std::vector<const char*> FilteringInfer::getInputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                     // Otherwise need vector<vector<>>
     
    //printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    //printf("Input %d : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    //printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    //for (int j = 0; j < input_node_dims.size(); j++)
    //  printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
    return input_node_names;    
}

 std::vector<const char*> FilteringInfer::getOutputNodes(Ort::Session& session){
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> output_node_dims; 

    //printf("Number of outputs = %zu\n", num_output_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_output_nodes; i++) {
    // print input node names
    char* output_name = session.GetOutputName(i, allocator);
    //printf("Input %d : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //printf("Output %d : type=%d\n", i, type);

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    //printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    //for (int j = 0; j < output_node_dims.size(); j++)
    //  printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    return output_node_names;    
}

std::vector<float> FilteringInfer::infer(Ort::Session& session,std::vector<float> input_tensor_values){
    //std::cout<<"got before";
    //__half *h_in;
    size_t input_tensor_size = input_tensor_values.size();
    //std::cout<<"Inf size "<<input_tensor_size<<"\n";
    //for (int i=0; i<input_tensor_size; i++)
    //    h_in[i] = __float2half(input_tensor_values[0]);
    //   std::cout << input_tensor_values[i] << " ";
    //std::cout<<"got after";
    //size_t free, total;
    //cudaMemGetInfo( &free, &total );
    //std::cout << " memory: free=" << free << ", total=" << total << "\n";
    
    //size_t input_tensor_size = input_tensor_values.size();
    std::vector<int64_t> input_node_dims(2); 
    input_node_dims[1] = 24;
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
    //auto memory_info = Ort::MemoryInfoCpu(OrtArenaAllocator, OrtMemTypeDefault);
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
    std::vector<float> output_tensor_values {floatarr, floatarr+input_node_dims[0]};
    //std::cout << "Output: " << output_tensor_values.size() << " ";
    return output_tensor_values;
}

bool FilteringInfer::teardown(){
    return true;
}
std::vector<float> FilteringInfer::processInput(long unsigned int& e_size){
    //std::ifstream in_file_e, in_file_f,in_file_fb;
    //read input data
    auto file_path = "datanmodels/in_e1000.csv";
    std::ifstream in_file_e (file_path);   /* open file */
    if (!in_file_e.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(file_path)).c_str());
    }
    std::string line;                    /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<float> data_x;      /* vector of vector<float> for 2d array */

    while (getline (in_file_e, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        data_x.insert (data_x.end(),row.begin(),row.end());  
    }
    in_file_e.close();
    
    file_path = "datanmodels/in_f1000.csv";
    std::ifstream in_file_f (file_path);   /* open file */
    if (!in_file_f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(file_path)).c_str());
    }
    //std::string line;                     /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<float> e_spatial_x;      /* vector of vector<float> for 2d array */

    while (getline (in_file_f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        e_spatial_x.insert (e_spatial_x.end(),row.begin(),row.end());  
    }
    in_file_f.close();

    //size_t data_size = data_x.size();
    long int data_size = data_x.size();
    //std::cout <<"Data Size " << data_size << "\n";
    torch::Tensor data_t = torch::from_blob((float *)data_x.data(),{data_size/12,12});//,torch::Device(torch::kCUDA, 0),dtype(torch::kFloat32));
    //std::cout << data_t.slice(0,0,100) << std::endl;
    long int e_spat_size = e_spatial_x.size();
    std::cout <<"Spatial Size " << e_spat_size << "\n";
    torch::Tensor e_spatial_t = torch::from_blob((float *)e_spatial_x.data(),{2,e_spat_size/2});
    //std::cout << e_spatial_t.slice(1,0,100) << "\n";
    e_spatial_t = e_spatial_t.transpose(1,0);
    //std::cout << e_spatial_t.slice(0,0,100) << "\n";
    
    auto v1 = e_spatial_t.index({Slice(),0}).to(torch::kLong);
    //std::cout << v1.slice(0,0,100) <<"\n";
    auto v2 = e_spatial_t.index({Slice(),1}).to(torch::kLong);
    //std::cout << "Here: "<< v2.slice(0,0,100) <<"\n";

    torch::Tensor sample_x = data_t.index({v1,Slice()});
    torch::Tensor sample_y = data_t.index({v2,Slice()});

    auto resc = torch::cat({sample_x, sample_y},1);
    std::cout << resc.slice(0,0,10) <<"\n";
    std::cout << resc.size(0) << " "<<resc.size(1)<< "\n";
    
    std::vector<float> data_l(resc.data_ptr<float>(), resc.data_ptr<float>() + resc.numel());
    std::cout << data_l.size() <<'\n';
    e_size = resc.size(0);

    return data_l;
}

bool FilteringInfer::verifyOutput(std::vector<float> output_tensor_values){
    size_t tensor_size = output_tensor_values.size();
    long unsigned int dim0 = tensor_size;
    for (unsigned int i = 0; i < dim0; i++)
        output_tensor_values[i] = 1.0 / (1.0 + exp(-output_tensor_values[i]));
    std::vector<std::size_t> shape = {dim0, 1};
    //auto output = xt::adapt(output_tensor_values, shape);
    for (unsigned int i = 0; i < 10; i++)
        std::cout << output_tensor_values[i] <<"\n";
    std::cout << dim0 <<" "<< 1 <<"\n";

    auto file_path = "datanmodels/out_f1000.csv";
    std::ifstream in_file_fb (file_path);   /* open file */
    if (!in_file_fb.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + std::string(file_path)).c_str());
    }
    std::string line;                    /* string to hold each line */
    //std::vector<std::vector<float>> array;      /* vector of vector<float> for 2d array */
    std::vector<float> e_spatial_b1;      /* vector of vector<float> for 2d array */

    while (getline (in_file_fb, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        e_spatial_b1.insert (e_spatial_b1.end(),row.begin(),row.end());  
    }
    in_file_fb.close();
    long int e_size = e_spatial_b1.size();
    torch::Tensor e_spatial_b1_t = torch::from_blob((float *)e_spatial_b1.data(),{e_size,1});   
    std::cout << e_spatial_b1_t.slice(0,0,10) << std::endl;
    
    //auto input = xt::load_csv<double>(in_file);
    //std::cout << input <<"\n";
    //auto shapeo = input.shape();   // shape = {2, 3}
    //std::cout << shapeo[0] << " " << shapeo[1] << "\n";
    
    
    //for (unsigned int i = 0; i < dim0; i++)
    //    std::cou
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
    
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        std::cout << "GPU " << id << "\n";
        std::cout << " memory: free=" << free << ", total=" << total << "\n";
    }
    
    std::cout << "Building and running a GPU inference engine for Filtering" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    FilteringInfer filter; //initializeSampleParams(args));
    
    Ort::Session session = filter.initOnnxSession(env);
    std::cout << "here:"<<"\n";
    
    cudaMemGetInfo( &free, &total );
    //std::cout << " memory: free=" << free << ", total=" << total << "\n";
    std::vector<int> e_spatial;
    long unsigned int e_size, batch_size=800000;

    std::vector<float> data = filter.processInput(e_size);
    //std::vector<std::size_t> shapen = {e_size, 24};
    //auto data_c = xt::adapt(data, shapen);

    
    int num_iter = e_size/batch_size;
    std::cout<< "E spatial: "<< num_iter << " " << e_size<< "\n";
    std::vector<float> output_f;
    std::vector<float> output_all;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < num_iter; i++) {
        std::vector<float> sample(data.begin()+i*batch_size*24, data.begin()+(i+1)*batch_size*24);
        output_f = filter.infer(session, sample);
        output_all.insert(output_all.end(), output_f.begin(), output_f.end());
    }
    std::vector<float> last(data.begin()+num_iter*batch_size*24, data.begin()+e_size*24);
    output_f = filter.infer(session, last);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl; 
    output_all.insert( output_all.end(), output_f.begin(),output_f.end() );

    bool res = filter.verifyOutput(output_all);
    std::cout << "Output correct:" << "true" << "\n";
    return 0;
}

