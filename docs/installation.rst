############
Installation
############

TorchFort can be installed in multiple ways but we highly recommend building and using a Docker container. 

Docker Installation
-------------------

We provide a `Dockerfile <https://github.com/NVIDIA/TorchFort/blob/master/docker/Dockerfile>`_ which contains all relevant dependencies and builds using the `NVIDIA HPC SDK <https://developer.nvidia.com/hpc-sdk>`_ software libraries and compilers, which is our recommended way to build TorchFort. In order to build TorchFort using Docker, simply clone the repo and call:

.. code-block:: bash

    docker build -t torchfort:latest -f docker/Dockerfile .

from the top level directory of the repo. Inside the container, TorchFort will be installed in ``/opt/torchfort``.

We provide an alternative docker file `Dockerfile_gnu <https://github.com/NVIDIA/TorchFort/blob/master/docker/Dockerfile_gnu>`_ which can be used to build TorchFort using GNU compilers. Additionally, we provide a docker file `Dockerfile_gnu_cpuonly <https://github.com/NVIDIA/TorchFort/blob/master/docker/Dockerfile_gnu_cpuonly>`_ which can be used to build TorchFort using GNU compilers without GPU support enabled.

CMake Installation
------------------

For a native installation TorchFort provides a `CMakeList.txt <https://github.com/NVIDIA/TorchFort/blob/master/CMakeLists.txt>`_ file. Please make sure that the following required packages are installed on your system before installing TorchFort:

* Requirements for core functionality and examples:

  - CUDA 12.1 or newer
  - ``python`` version 3.6 or higher
  - ``pybind11``
  - ``yaml-cpp`` from https://github.com/jbeder/yaml-cpp.git
  - MPI
  - NVIDIA Collective Communication Library (``NCCL``)
  - ``HDF5``
  - the Python modules specified in `requirements.txt <https://github.com/NVIDIA/TorchFort/blob/master/requirements.txt>`_
  - GNU or `NVHPC <https://developer.nvidia.com/hpc-sdk>`_ compilers. NVHPC compilers are **required** if CUDA Fortran device array support is desired.

* Additional requirements for building this documentation:

  - Doxygen
  - the Python modules specified in `docs/requirements.txt <https://github.com/NVIDIA/TorchFort/blob/master/docs/requirements.txt>`_

For CPU-only builds, CUDA and NCCL are not required.


To build TorchFort, clone the repo then call the following from the root directory:

.. code-block:: bash

    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=<TorchFort installation prefix> \
          -DTORCHFORT_YAML_CPP_ROOT=<path to yaml-cpp installation> \
          -DTORCHFORT_BUILD_EXAMPLES=1 \
          -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`" \
        ..
    make -j install

See the top level `CMakeList.txt <https://github.com/NVIDIA/TorchFort/blob/master/CMakeLists.txt>`_ file for additional CMake configuration options.
    
Build Documentation
-------------------

The documentation can be built with the corresponding ``Makefile`` in the ``docs`` directory. Make sure that the requirements are installed and call:

.. code-block:: bash

    cd docs && make html

The docs will be located in ``docs/_build/html`` and can be viewed locally in your web browser. 

Directory Structure
-------------------

Independent of how you decide to install TorchFort, the directory structure will be as follows::

    <TorchFort installation prefix>
    |--- bin
         |--- examples
              |--- cpp
              |--- fortran
         |--- python
    |--- include
    |--- lib
    
The ``bin`` folder contains the examples written in C++ or Fortran located in the corresponding subdirectories. The ``python`` subfolder contains the Python wrappers for :ref:`wandb_support-ref`.

The Fortran module ``torchfort.mod`` as well as the C headers can be found inside the ``include`` folder and the dynamic libraries inside the ``lib`` folder.
