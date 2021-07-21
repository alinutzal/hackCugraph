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

project(CUGRAPH_TESTS VERSION 0.1.0 LANGUAGES CXX CUDA)

#set(CUDA_ARCHITECTURES 75)

# find_package(cuml)

#find_package(rmm)
# include(cmake/CPM.cmake)
# CPMAddPackage(
#   NAME cuco
#   GITHUB_REPOSITORY NVIDIA/cuCollections
#   GIT_TAG dev
#   OPTIONS
#      "BUILD_TESTS OFF"
#      "BUILD_BENCHMARKS OFF"
#      "BUILD_EXAMPLES OFF"
# )

# set(CUGRAPH_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/libcugraph-21.06.00-cuda11.0_gf9ffd2de_0/include)
# set(CUML_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/libcuml-21.06.02-cuda11.0_g7dfbf8d9e_0/include)
# set(RMM_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/librmm-21.06.00-cuda11.0_gee432a0_0/include)
# set(SLOG_INCLUDE_DIRS /home/alinal/miniconda3/pkgs/spdlog-1.8.5-h4bd325d_0/include)

add_subdirectory(src)