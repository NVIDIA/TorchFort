FROM  nvcr.io/nvidia/nvhpc:25.7-devel-cuda12.9-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV CUDA_HOME /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install System Dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y && \
    apt install -y wget cmake && \
    apt install -y python3 python-is-python3 python3-pip python3-pybind11 && \
    apt install -y vim gfortran

# Install PyTorch (and select dependencies)
RUN pip3 install --no-deps torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129 && \
    pip3 install --no-deps nvidia-cudnn-cu12==9.10.2.21 nvidia-cusparselt-cu12==0.7.1 nvidia-cufile-cu12==1.14.1.1 nvidia-nccl-cu12==2.27.3

# Remove conflicting NCCL version from NVHPC SDK, add libnccl.so symlink to pip installed NCCL
RUN rm -rf /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/nccl/ && \
    cd /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib && \
    ln -s libnccl.so.2 libnccl.so
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# Install yaml-cpp
RUN git clone https://github.com/jbeder/yaml-cpp.git --branch 0.8.0 && \
    cd yaml-cpp && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/yaml-cpp \
          -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
          -DBUILD_SHARED_LIBS=OFF \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    make -j$(nproc) && make install
ENV LD_LIBRARY_PATH /opt/yaml-cpp/lib:${LD_LIBRARY_PATH}

# Install additional Python dependencies
RUN pip3 install wandb ruamel-yaml matplotlib pygame moviepy

# Install TorchFort
ENV FC=nvfortran
COPY . /torchfort
RUN cd /torchfort && mkdir build && cd build && \
    CUDA_PATH=$CUDA_HOME \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/torchfort \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DTORCHFORT_YAML_CPP_ROOT=/opt/yaml-cpp \
    -DTORCHFORT_NCCL_ROOT=/usr/local/lib/python3.10/dist-packages/nvidia/nccl \
    -DTORCHFORT_BUILD_EXAMPLES=1 \
    -DTORCHFORT_BUILD_TESTS=1 \
    -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`" \
    .. && \
    make -j$(nproc) install && \
    cd / && rm -rf torchfort
ENV LD_LIBRARY_PATH /opt/torchfort/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}
