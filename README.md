# TorchFort

An Online Deep Learning Interface for HPC programs on NVIDIA GPUs

## Introduction

TorchFort is a DL training and inference interface for HPC programs implemented using LibTorch, the C++ backend used by the [PyTorch](https://pytorch.org]) framework.
The goal of this library is to help practitioners and domain scientists to seamlessly combine their simulation codes with Deep Learning functionalities available
within PyTorch.
This library can be invoked directly from Fortran or C/C++ programs, enabling transparent sharing of data arrays to and from the DL framework all contained within the
simulation process (i.e., no external glue/data-sharing code required). The library can directly load PyTorch model definitions exported to TorchScript and implements a
configurable training process that users can control via a simple YAML configuration file format. The configuration files enable users to specify optimizer and loss selection,
learning rate schedules, and much more.

Please refer to the [documentation](https://nvidia.github.io/TorchFort/) for additional information on the library, build instructions, and usage details.

Please refer to the [examples](examples) to see TorchFort in action.

Contact us or open a GitHub issue if you are interested in using this library in your own solvers and have questions on usage and/or feature requests.

## Extensions

The library has been extended with the following features:

- report of the bytes exchanged between ranks and time taken
- Unified Memory support and tuning for train_distributed program

Missing features:

- oversubscription was not implemented due to train_distributed being an OpenAcc program

## Future work

- Replace CUDA-Aware MPI with NCCL for better performance and less page faults for Unified Memory

## Installation

The library is built using CMake and requires the following dependencies:

- [LibTorch](https://pytorch.org/get-started/locally/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/)

To build the library on Polaris, where this extension to the library was tested refer to the [installer](polaris_install.sh)

## License

This library is released under a BSD 3-clause license, which can be found in [LICENSE](license).
