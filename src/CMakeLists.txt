# find_package(CUDA)
# find_package(CUDAToolkit)
# find_package(cuml) # ML:DBScan
# find_package(cugraph) # for graph
# find_package(cuco)


include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# message(STATUS "Cugraph DIR" ${CUGRAPH_INCLUDE_DIRS})
# add_executable(weakly_connected_components_test weakly_connected_components_test.cpp)

# target_link_libraries(weakly_connected_components_test
# PUBLIC
#     cugraph
# )
# if (NOT CONDA_ENV_DIR)
#     message(WARNING "Need CONDA_ENV_DIR to figure out dep., set to /opt/conda/envs/rapids")
#     set(CONDA_ENV_DIR /opt/conda/envs/rapids)
# endif ()

add_executable(mmio_read mmio_read.cpp 
        mmio/mmio.c 
        matrix_market_file_utilities.cu
        misc_utilities.cpp
)

set_target_properties(mmio_read
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
            CUDA_STANDARD                       17
            CUDA_STANDARD_REQUIRED              ON
            INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(mmio_read
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(mmio_read PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_include_directories(mmio_read
PUBLIC
    ${CONDA_ENV_DIR}/include
    # /rapids/cugraph/cpp/include

    ${CUCO_INCLUDE_DIR}
)
target_link_libraries(mmio_read
PUBLIC
    # ${CONDA_ENV_DIR}/lib/libcugraph${CMAKE_SHARED_LIBRARY_SUFFIX}
    $<BUILD_INTERFACE:conda_env>
    cugraph
    cuco::cuco
    # NCCL::NCCL
)
# Add Conda env exist s, use it for link and include dirs
# if(TARGET conda_env)
#   target_link_libraries(mmio_read PUBLIC $<BUILD_INTERFACE:conda_env> )
# endif()


# add_executable(dbscan_test cugraphExample.cu)

# # set_target_properties(cugraph_example
# #     PROPERTIES CXX_STANDARD                        17
# #                CXX_STANDARD_REQUIRED               ON
# #                CUDA_STANDARD                       17
# #                CUDA_STANDARD_REQUIRED              ON
# #                INTERFACE_POSITION_INDEPENDENT_CODE ON
# # )


# target_include_directories(cugraph_example 
#     PRIVATE 
#     ${CUB_INCLUDE_DIR}
#     ${CUCO_INCLUDE_DIR}
#     ${CUML_INCLUDE_DIRS} 
#     ${CUGRAPH_INCLUDE_DIRS} 
#     ${CUDA_INCLUDE_DIRS}  
#     ${RMM_INCLUDE_DIRS} 
#     ${SLOG_INCLUDE_DIRS})
# set_target_properties(cugraph_example PROPERTIES LINKER_LANGUAGE "CUDA")
# target_link_libraries(cugraph_example cuml++ cudart cublas cusolver cusparse cugraph cuco)