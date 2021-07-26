
#include <utilities/test_graphs.hpp>
#include <utilities/high_res_clock.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/experimental/graph.hpp>
#include <cugraph/experimental/graph_functions.hpp>
#include <cugraph/experimental/graph_view.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
    cugraph::test::File_Usecase const & input_usecase = cugraph::test::File_Usecase("karate.mtx");

    constexpr bool renumber = true;

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;

    static int PERF = 0;

    raft::handle_t handle{};
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
    
    auto graph_view = graph.view();

    rmm::device_uvector<vertex_t> d_components(graph_view.get_number_of_vertices(),
                                                      handle.get_stream());

    cugraph::experimental::weakly_connected_components(
          handle, graph_view, d_components.data());


    return 0;
}