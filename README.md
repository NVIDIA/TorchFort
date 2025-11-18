# TorchFort

An Online Deep Learning Interface for HPC programs on NVIDIA GPUs

## Introduction
TorchFort is a DL training and inference interface for HPC programs implemented using LibTorch, the C++ backend used by the [PyTorch](https://pytorch.org) framework.
The goal of this library is to help practitioners and domain scientists to seamlessly combine their simulation codes with Deep Learning functionalities available 
within PyTorch.
This library can be invoked directly from Fortran or C/C++ programs, enabling transparent sharing of data arrays to and from the DL framework all contained within the
simulation process (i.e., no external glue/data-sharing code required). The library can directly load PyTorch model definitions exported to TorchScript and implements a
configurable training process that users can control via a simple YAML configuration file format. The configuration files enable users to specify optimizer and loss selection,
learning rate schedules, and much more.

Please refer to the [documentation](https://nvidia.github.io/TorchFort/) for additional information on the library, build instructions, and usage details.

Please refer to the [examples](examples) to see TorchFort in action.

## How to Use TorchFort

TorchFort provides a simple and powerful interface for integrating deep learning capabilities into HPC simulation codes written in Fortran, C, or C++. The library supports both **supervised learning** and **reinforcement learning** paradigms.

### Quick Start

1. **Create a Model**: Initialize a TorchFort model using a YAML configuration file
   ```fortran
   istat = torchfort_create_model("my_model", "config.yaml", device_id)
   ```

2. **Train the Model**: Provide input data and labels from your simulation
   ```fortran
   istat = torchfort_train("my_model", input_array, label_array, loss_value, stream)
   ```

3. **Run Inference**: Generate predictions using your trained model
   ```fortran
   istat = torchfort_inference("my_model", input_array, output_array, stream)
   ```

4. **Save/Load Checkpoints**: Preserve your training progress
   ```fortran
   istat = torchfort_save_checkpoint("my_model", "checkpoint_dir")
   istat = torchfort_load_checkpoint("my_model", "checkpoint_dir", step_train, step_inference)
   ```

### Key Features

- **Direct Integration**: Call TorchFort functions directly from Fortran, C, or C++ code
- **Zero-Copy Data Sharing**: Efficient data transfer between simulation and ML framework
- **GPU Acceleration**: Native support for NVIDIA GPUs with CUDA stream integration
- **Flexible Model Definition**: Use built-in models or load custom PyTorch models via TorchScript
- **YAML Configuration**: Control training parameters, optimizers, learning rates, and more through simple configuration files
- **Multi-GPU Support**: Data-parallel training across multiple GPUs
- **OpenACC Compatible**: Seamlessly integrate with OpenACC-accelerated codes

### Use Cases

TorchFort is designed for HPC applications including:
- **Weather and Climate Modeling**: Integrate ML into atmospheric simulations (e.g., WRF, MPAS)
- **Computational Fluid Dynamics**: Learn closure models, turbulence parameterizations
- **Online Training**: Train models on-the-fly using data generated during simulation
- **Surrogate Modeling**: Replace expensive computation with fast ML approximations
- **Reinforcement Learning**: Optimize control policies for complex physical systems

For detailed integration guides with specific HPC applications, see:
- [WRF Integration Guide](docs/WRF_INTEGRATION.md) - Using TorchFort with the Weather Research and Forecasting model
- [MPAS Integration Guide](docs/MPAS_INTEGRATION.md) - Using TorchFort with the Model for Prediction Across Scales

Contact us or open a GitHub issue if you are interested in using this library in your own solvers and have questions on usage and/or feature requests.

## License
This library is released under an Apache 2.0 license, which can be found in [LICENSE](LICENSE).
