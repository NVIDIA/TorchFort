FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

# Install System Dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt update -y && \
    apt install -y curl unzip wget cmake && \
    apt install -y python3 python-is-python3 python3-pip python3-pybind11 && \
    apt install -y git vim gfortran doxygen && \
    apt install -y libibverbs-dev ibverbs-utils numactl

# Install NVHPC SDK
RUN wget https://developer.download.nvidia.com/hpc-sdk/25.3/nvhpc_2025_253_Linux_x86_64_cuda_12.8.tar.gz && \
    tar xpzf nvhpc_2025_253_Linux_x86_64_cuda_12.8.tar.gz && \
    nvhpc_2025_253_Linux_x86_64_cuda_12.8/install --quiet && \
    rm -rf nvhpc_2025_253_Linux_x86_64_cuda_12.8 nvhpc_2025_253_Linux_x86_64_cuda_12.8.tar.gz

ENV PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/bin:$PATH
ENV PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/nvshmem/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda

RUN echo "source /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/latest/hpcx-init.sh; hpcx_load" >>  /root/.bashrc

# Install PyTorch
RUN pip3 install torch==2.7.0

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

# Install HDF5
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz && \
    tar xzf hdf5-1_14_3.tar.gz && \
    cd hdf5-hdf5-1_14_3 && \
    CC=mpicc FC=mpifort FCFLAGS=-fPIC CFLAGS=-fPIC \
    ./configure --enable-parallel \
                --enable-fortran \
                --prefix=/opt/hdf5 && \
    make -j$(nproc) install && \
    cd .. && \
    rm -rf hdf5-hdf5-1_14_3 hdf5-1_14_3.tar.gz
ENV LD_LIBRARY_PATH /opt/hdf5/lib:$LD_LIBRARY_PATH

# Install additional Python dependencies
RUN pip3 install wandb ruamel-yaml h5py matplotlib pygame moviepy

# Install TorchFort
ENV FC=nvfortran
ENV HDF5_ROOT=/opt/hdf5
COPY . /torchfort
RUN cd /torchfort && mkdir build && cd build && \
    CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/torchfort \
    -DCMAKE_CXX_COMPILER=`which g++` \
    -DTORCHFORT_YAML_CPP_ROOT=/opt/yaml-cpp \
    -DTORCHFORT_NCCL_ROOT=/opt/nccl/build \
    -DTORCHFORT_BUILD_EXAMPLES=1 \
    -DTORCHFORT_BUILD_TESTS=1 \
    -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`" \
    .. && \
    make -j$(nproc) install && \
    cd / && rm -rf torchfort
ENV LD_LIBRARY_PATH /opt/torchfort/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

ENTRYPOINT bash
