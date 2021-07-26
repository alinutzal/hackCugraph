# ONNX Runtime Inference

## Introduction

ONNX Runtime C++ inference example for the ExaTrkX pipeline using CPU and CUDA.

## Dependencies

* CMake 3.17.2
* ONNX Runtime 1.7.0 (include and library) follow instructions on [Setting up ONNX Runtime on Ubuntu 20.04 (C++ API)] (https://stackoverflow.com/questions/63420533/setting-up-onnx-runtime-on-ubuntu-20-04-c-api)
* CUDA 11.0.3
* a conda environment with xtensor, xlt, faiss, cudf and cugraph

## Modules

* module load PrgEnv-intel/6.0.9
* module load cmake/3.20.5
* module load cuda/11.0.3
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

export CC=/apps/gnu/9.1.0/bin/gcc
export CXX=/apps/gnu/9.1.0/bin/g++
### Build Example

```bash
$ cd exatrkx-work/Inference
$ cmake -B build -DCMAKE_PREFIX_PATH=/users/PLS0129/ysu0053/libtorch
$ cmake -B build
$ make -C build 1_embed
$ make -C build 2_5_build_edge
$ make -C build 3_filtering
$ 
```

### Run Example

```bash
$ ./build/1_embed --use_cpu
$ ./build/2_build_edge --use_cpu
$ ./build/inference  --use_cpu
Inference Execution Provider: CPU
```

```bash
$ ./build/1_embed --use_cuda
$ ./build/2_build_edge --use_cuda
$ ./build/3_filtering --use_cuda
$ ./build/inference  --use_cuda
Inference Execution Provider: CUDA
```

## References

* 