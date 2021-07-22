include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_executable(mmio_read mmio_read.cpp 
        mmio/mmio.c 
        matrix_market_file_utilities.cu
        misc_utilities.cpp
)

set_target_properties(mmio_read
PROPERTIES  BUILD_RPATH    "\$ORIGIN"
            INSTALL_RPATH  "\$ORIGIN"
            CXX_STANDARD                        17
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


target_link_libraries(mmio_read
PUBLIC
    # ${CONDA_ENV_DIR}/lib/libcugraph${CMAKE_SHARED_LIBRARY_SUFFIX}
    $<BUILD_INTERFACE:conda_env>
    cugraph
    cuco::cuco
)