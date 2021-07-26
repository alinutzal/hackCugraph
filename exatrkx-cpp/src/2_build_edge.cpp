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

//#include <faiss/IndexFlat.h>
//#include <faiss/gpu/GpuIndexFlat.h>
//#include <faiss/gpu/GpuIndexIVFFlat.h>
//#include <faiss/gpu/StandardGpuResources.h>

using namespace xt::placeholders;  // required for `_` to work

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

    std::ifstream in_file, out_file;
    in_file.open("datanmodels/out_e1000.csv");
    auto spatial = xt::load_csv<double>(in_file);
    std::cout << "Spatial:\n"<<spatial <<"\n";

    out_file.open("datanmodels/out_b1000.csv");
    auto e_spatial = xt::load_csv<float>(out_file);
    std::cout << "E-Spatial:\n"<<e_spatial <<"\n";
    
    std::vector<float> input_tensor_values(spatial.begin(), spatial.end());
    size_t input_tensor_size = input_tensor_values.size();
    
    std::vector<std::size_t> shape = { input_tensor_size/12, 8 };
    std::cout << shape[0] << " " <<shape[1]<<"\n";
    
    float* floatarr = &input_tensor_values[0];
    
    int r_max = 1.7;
    int k_max = 500;
    int d = shape[1];
    //iss::IndexFlatL2 index_flat(d);   
    if (useCUDA) {
        int nq = shape[0];
        faiss::gpu::StandardGpuResources res;
        // Using a flat index
        faiss::gpu::GpuIndexFlatL2 gpu_index_flat(&res, d);
        gpu_index_flat.add(nq, floatarr); // add vectors to the index
        printf("is_trained = %s\n", gpu_index_flat.is_trained ? "true" : "false");
        printf("ntotal = %ld\n", gpu_index_flat.ntotal);
        { // search xq
        long* I = new long[k_max * nq];
        float* D = new float[k_max * nq];

        gpu_index_flat.search(nq, floatarr, k_max, D, I);

        /* print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k_max; j++)
                printf("%5ld ", I[i * k_max + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k_max; j++)
                printf("%5ld ", I[i * k_max + j]);
            printf("\n");
        }
/*
        ind = torch.Tensor.repeat(torch.arange(I.shape[0]), (I.shape[1], 1), 1).T.to(device)
        edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])
        if return_indices:
            return edge_list, D, I, ind
        else:
            return edge_list
*/            
        delete[] I;
        delete[] D;
    }

    }

}