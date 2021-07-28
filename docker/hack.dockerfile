FROM rapidsai/rapidsai-dev:21.06-cuda11.0-devel-ubuntu20.04-py3.8
#FROM nvcr.io/nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# FROM nvcr.io/nvidia/tensorrt:20.09-py3

ARG ONNXRUNTIME_VERSION=1.8.0
ARG LIBTORCH_VERSION=1.9.0
ARG NUM_JOBS=12

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN cd ../home
RUN wget https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.9.0%2Bcu111.zip \
&& unzip libtorch-shared-with-deps-1.9.0+cu111.zip && rm libtorch-shared-with-deps-1.9.0+cu111.zip

RUN mkdir /tmp/onnxInstall && cd /tmp/onnxInstall
RUN wget -O onnx_archive.nupkg https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.8.0 \
&& unzip onnx_archive.nupkg && rm onnx_archive.nupkg

RUN mkdir -p /home/onnx/lib && mkdir -p /home/onnx/include/onnxruntime/
RUN cp runtimes/linux-x64/native/libonnxruntime.so /home/onnx/lib/ \
&& cp -r build/native/include/ /home/onnx/include/onnxruntime/
RUN chmod -R a+rX /home/onnx/