#!bin/bash

# Since CUDA Aware-MPI is being used the following env must be set and module loaded
export MPICH_GPU_SUPPORT_ENABLED=1 && \
module load craype-accel-nvidia80 && \
cmake -DCMAKE_INSTALL_PREFIX=/home/tartarughina/TorchFort-def \
-DTORCHFORT_BUILD_EXAMPLES=1 \
-DCMAKE_PREFIX_PATH=/soft/libraries/libtorch/libtorch-2.4.0+cu124/share/cmake \
..

-DTORCHFORT_YAML_CPP_ROOT=/home/tartarughina/yaml-cpp-def \
-DCMAKE_PREFIX_PATH="`python3.11 -c 'import torch;print(torch.utils.cmake_prefix_path)'`" \

export LD_LIBRARY_PATH=/soft/libraries/libtorch/libtorch-2.4.0+cu124/lib:$HOME/hdf5/build/HDF5-1.17.0-Linux/HDF_Group/HDF5/1.17.0/lib:$HOME/TorchFort-def/lib:$LD_LIBRARY_PATH && \
export MPICH_GPU_SUPPORT_ENABLED=1 && \
module load craype-accel-nvidia80
