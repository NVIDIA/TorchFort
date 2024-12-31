#!bin/bash

# Since CUDA Aware-MPI is being used the following env must be set and module loaded
export MPICH_GPU_SUPPORT_ENABLED=1 && \
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1 && \
export MPICH_RDMA_ENABLED_CUDA=1 && \
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH && \
module load craype-accel-nvidia80 && \
module load cray-hdf5/1.12.2.9 && \
cmake -DCMAKE_INSTALL_PREFIX=/home/tartarughina/TorchFort-def \
-DTORCHFORT_BUILD_EXAMPLES=1 \
-DCMAKE_PREFIX_PATH=/soft/libraries/libtorch/libtorch-2.4.0+cu124/share/cmake \
..


# In order to run the program the following commands need to be executed
export LD_LIBRARY_PATH=/soft/libraries/libtorch/libtorch-2.4.0+cu124/lib:$HOME/TorchFort-def/lib:$LD_LIBRARY_PATH && \
export MPICH_GPU_SUPPORT_ENABLED=1 && \
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1 && \
export MPICH_RDMA_ENABLED_CUDA=1 && \
module load craype-accel-nvidia80 && \
module load cray-hdf5/1.12.2.9
