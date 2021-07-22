#=============================================================================
#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=============================================================================

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

add_library(cugraphtestutil STATIC
            utilities/matrix_market_file_utilities.cu
            utilities/rmat_utilities.cu
            utilities/thrust_wrapper.cu
            utilities/misc_utilities.cpp
            components/wcc_graphs.cu
            mmio/mmio.c)

target_compile_options(cugraphtestutil
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                   "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>>"
)

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphtestutil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/mmio"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        # "${CUGRAPH_SOURCE_DIR}/src"
)

target_link_libraries(cugraphtestutil
    PUBLIC
        $<BUILD_INTERFACE:conda_env>
        cugraph
        cuco::cuco
        NCCL::NCCL
)


# add_library(cugraphmgtestutil STATIC
#             "${CMAKE_CURRENT_SOURCE_DIR}/utilities/device_comm_wrapper.cu")

# set_property(TARGET cugraphmgtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

# target_include_directories(cugraphmgtestutil
#     PRIVATE
#     "${CUB_INCLUDE_DIR}"
#     "${THRUST_INCLUDE_DIR}"
#     "${CUCO_INCLUDE_DIR}"
#     "${LIBCUDACXX_INCLUDE_DIR}"
#     "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
#     "${RMM_INCLUDE}"
#     "${NCCL_INCLUDE_DIRS}"
#     "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
#     "${CMAKE_CURRENT_SOURCE_DIR}/../include"
#     "${CMAKE_CURRENT_SOURCE_DIR}"
#     "${RAFT_DIR}/cpp/include"
# )

# target_link_libraries(cugraphmgtestutil cugraph)


include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

add_executable(mmio_read mmio_read.cpp)

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
    # # ${CONDA_ENV_DIR}/lib/libcugraph${CMAKE_SHARED_LIBRARY_SUFFIX}
    # cuco::cuco
    cugraphtestutil
    $<BUILD_INTERFACE:conda_env>
    cugraph
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    CUDA::curand
)