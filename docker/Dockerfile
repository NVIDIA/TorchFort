FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# install apt dependencies
RUN apt update -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y curl unzip wget cmake python3 python-is-python3 python3-pip python3-pybind11 git vim gfortran doxygen

# install nvhpc:
RUN wget https://developer.download.nvidia.com/hpc-sdk/23.5/nvhpc_2023_235_Linux_x86_64_cuda_multi.tar.gz && \
    tar xpzf nvhpc_2023_235_Linux_x86_64_cuda_multi.tar.gz && \
    nvhpc_2023_235_Linux_x86_64_cuda_multi/install --quiet && \
    rm -rf nvhpc_2023_235_Linux_x86_64_cuda_multi nvhpc_2023_235_Linux_x86_64_cuda_multi.tar.gz

ENV PATH ${PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/compilers/bin
ENV PATH ${PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/mpi/bin
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/mpi/lib
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/11.8/nccl/lib
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/comm_libs/11.8/nvshmem/lib
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/2023/math_libs/11.8/lib64
ENV NVHPC_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cuda/11.8

# install python stuff
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# install yaml-cpp
RUN git clone https://github.com/jbeder/yaml-cpp.git --branch yaml-cpp-0.7.0 && \
    cd yaml-cpp && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/yaml-cpp -DCMAKE_CXX_FLAGS:="-D_GLIBCXX_USE_CXX11_ABI=0" -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    make -j && make install
ENV LD_LIBRARY_PATH /opt/yaml-cpp/lib:${LD_LIBRARY_PATH}

# install hdf5
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_3.tar.gz && \
    tar xzf hdf5-1_13_3.tar.gz && \
    cd hdf5-hdf5-1_13_3 && \
    CC=mpicc FC=mpifort FCFLAGS=-fPIC CFLAGS=-fPIC ./configure --enable-parallel --enable-fortran --prefix=/opt/hdf5 && \
    make -j install && \
    cd .. && \
    rm -rf hdf5-hdf5-1_13_3 hdf5-1_13_3.tar.gz
ENV LD_LIBRARY_PATH /opt/hdf5/lib:${LD_LIBRARY_PATH}

# install additional python packages
RUN pip3 install wandb ruamel-yaml h5py matplotlib pygame moviepy

# install torchfort
ENV FC=nvfortran
ENV HDF5_ROOT=/opt/hdf5
COPY . /torchfort
RUN cd /torchfort && mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/torchfort \
    -DNVHPC_CUDA_VERSION=11.8 \
    -DYAML_CPP_ROOT=/opt/yaml-cpp \
    -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`;/opt/nvidia/hpc_sdk/Linux_x86_64/23.5/cmake" \
    .. && \
    make -j install && \
    cd / && rm -rf torchfort
ENV LD_LIBRARY_PATH /opt/torchfort/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/lib/python3.8/dist-packages/torch/lib:${LD_LIBRARY_PATH}
