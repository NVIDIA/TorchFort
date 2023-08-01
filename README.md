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

## License
This library is released under a BSD 3-clause license, which can be found in [LICENSE](license).
