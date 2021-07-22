#=============================================================================
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#=============================================================================
cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
include(FetchContent)
FetchContent_Declare(
  rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake.git
  GIT_TAG        origin/branch-21.06
  )
FetchContent_MakeAvailable(rapids-cmake)

include(cmake/CPM.cmake)
include(rapids-cpm)
include(rapids-cmake)
include(rapids-cuda)
# include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH_TESTS)

project(CUGRAPH_TESTS VERSION 0.1.0 LANGUAGES CXX CUDA)

# Remove the following archs from CMAKE_CUDA_ARCHITECTURES that
# cuhornet currently doesn't support
#
# < 60
# >= 86
set(supported_archs "60" "62" "70" "72" "75" "80")
foreach( arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE "-real" "" arch ${arch})
    if( arch IN_LIST supported_archs )
        list(APPEND usable_arch_values ${arch})
    endif()
endforeach()
# Make sure everything but the 'newest' arch
# is marked as `-real` so we only generate PTX for
# arch > 80
list(POP_BACK usable_arch_values latest_arch)
list(TRANSFORM usable_arch_values APPEND "-real")
list(APPEND usable_arch_values ${latest_arch})

set(CMAKE_CUDA_ARCHITECTURES ${usable_arch_values})

CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG 0b672bbde7c85a79df4d7ca5f82e15e5b4a57700
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-exports
    INSTALL_EXPORT_SET  cugraph-exports
    )

set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND CUGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


###################################################################################################
# - find blas -------------------------------------------------------------------------------------

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------


rapids_cpm_init()
include(cmake/get_cuco.cmake)

# include(cmake/get_raft.cmake)

# set(CUGRAPH_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/libcugraph-21.06.00-cuda11.0_gf9ffd2de_0/include)
# set(CUML_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/libcuml-21.06.02-cuda11.0_g7dfbf8d9e_0/include)
# set(RMM_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/librmm-21.06.00-cuda11.0_gee432a0_0/include)
# set(SLOG_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/spdlog-1.8.5-h4bd325d_0/include)

add_subdirectory(src)