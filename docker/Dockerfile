FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

# Install System Dependencies
RUN apt update -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y curl unzip wget cmake python3 python-is-python3 python3-pip python3-pybind11 git vim gfortran doxygen libibverbs-dev

# Install NVHPC SDK
RUN wget https://developer.download.nvidia.com/hpc-sdk/24.1/nvhpc_2024_241_Linux_x86_64_cuda_12.3.tar.gz && \
    tar xpzf nvhpc_2024_241_Linux_x86_64_cuda_12.3.tar.gz && \
    nvhpc_2024_241_Linux_x86_64_cuda_12.3/install --quiet && \
    rm -rf nvhpc_2024_241_Linux_x86_64_cuda_12.3 nvhpc_2024_241_Linux_x86_64_cuda_12.3.tar.gz

ENV PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/compilers/bin:$PATH
ENV PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/nvshmem/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda

RUN echo "source /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/comm_libs/12.3/hpcx/latest/hpcx-init.sh; hpcx_load" >>  /root/.bashrc

# Install NCCL 2.20.3 for compatibility with PyTorch 2.2.1
RUN cd /opt && \
    git clone --branch v2.20.3-1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90" CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda
ENV LD_LIBRARY_PATH /opt/nccl/build/lib:$LD_LIBRARY_PATH

# Install PyTorch
RUN pip3 install torch==2.2.1 torchvision torchaudio

# Install yaml-cpp
RUN git clone https://github.com/jbeder/yaml-cpp.git --branch 0.8.0 && \
    cd yaml-cpp && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/yaml-cpp \
          -DCMAKE_CXX_FLAGS:="-D_GLIBCXX_USE_CXX11_ABI=0" \
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
    CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/torchfort \
    -DTORCHFORT_YAML_CPP_ROOT=/opt/yaml-cpp \
    -DTORCHFORT_NCCL_ROOT=/opt/nccl/build \
    -DTORCHFORT_BUILD_EXAMPLES=1 \
    -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`" \
    .. && \
    make VERBOSE=1 -j$(nproc) install && \
    cd / && rm -rf torchfort
ENV LD_LIBRARY_PATH /opt/torchfort/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}

ENTRYPOINT bash
