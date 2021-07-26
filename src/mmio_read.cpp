
#include <utilities/test_graphs.hpp>
#include <utilities/high_res_clock.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>

#include <iostream>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call,                                                          \
              __LINE__,                                                       \
              __FILE__,                                                       \
              cudaGetErrorString(cudaStatus),                                 \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL

int main(int argc, char* argv[])
{
    raft::handle_t handle;
    std::shared_ptr<raft::mr::device::allocator> allocator(new raft::mr::device::default_allocator());
    handle.set_device_allocator(allocator);
    
    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    

    
    cugraph::test::File_Usecase const & input_usecase = cugraph::test::File_Usecase("netscience.mtx");

    constexpr bool renumber = true;

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    static int PERF = 0;

    //raft::handle_t handle{};
    HighResClock hr_clock{};

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }

    // the last two booleans are:
    // store_transposed and multi-gpu
    cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);
    rmm::device_uvector<vertex_t> d_renumber_map_labels(0, handle.get_stream());
    std::tie(graph, d_renumber_map_labels) =
      input_usecase.template construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, false, renumber);

    auto graph_view = graph.view();
    CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

    std::cout << "Number of Nodes:" << graph_view.get_number_of_vertices() << std::endl;
    std::cout << "Number of Edges:" << graph_view.get_number_of_edges() << std::endl;

    int num_edges = graph_view.get_number_of_edges();
    int num_vert = graph_view.get_number_of_vertices();
    
    rmm::device_uvector<vertex_t> d_components(graph_view.get_number_of_vertices(),
                                                      handle.get_stream());

    cugraph::experimental::weakly_connected_components(handle, graph_view, d_components.data());
    
    std::vector<int> h_labels(num_vert);
    
    CUDA_RT_CALL(cudaMemcpyAsync(h_labels.data(), d_components.data(), num_vert * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    int nRows = num_vert;
      std::map<long, size_t> histogram;
      for (int row = 0; row < nRows; row++) {
        std::cout << std::setw(10) << h_labels[row] << std::endl;
        if (histogram.find(h_labels[row]) == histogram.end()) {
          histogram[h_labels[row]] = 1;
        } else {
          histogram[h_labels[row]]++;
        }
      }

      size_t nClusters = 0;
      size_t noise     = 0;
      std::cout << "Histogram of samples" << std::endl;
      std::cout << "Cluster id, Number samples" << std::endl;
      for (auto it = histogram.begin(); it != histogram.end(); it++) {
        if (it->first != -1) {
          std::cout << std::setw(10) << it->first << ", " << it->second << std::endl;
          nClusters++;
        } else {
          noise += it->second;
        }
      }

      std::cout << "Total number of clusters: " << nClusters << std::endl;
      std::cout << "Noise samples: " << noise << std::endl;

      //CUDA_RT_CALL(cudaFree(d_components));
      //CUDA_RT_CALL(cudaFree(d_inputData));
      CUDA_RT_CALL(cudaStreamDestroy(stream));
      CUDA_RT_CALL(cudaDeviceSynchronize());

    return 0;
}