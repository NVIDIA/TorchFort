Installation
============

TorchFort can be installed in multiple ways but we highly recommend building and using a Docker container. 

Docker Installation
-------------------

We provide a ``Dockerfile`` which contains all relevant dependencies. In order to build TorchFort using Docker, simply clone the repo and call:

.. code-block:: bash

    docker build -t torchfort:latest -f docker/Dockerfile .

from the top level directory of the repo. Inside the container, TorchFort will be installed in ``/opt/torchfort``.

CMake Installation
------------------

For a native installation TorchFort provides a ``CMakeList.txt`` file. Please make sure that the following required packages are installed on your system before installing TorchFort:

* Requirements for core functionality and examples:

  - CUDA 11.8 (CUDA 12.x is **not** supported yet)
  - ``python`` version 3.6 or higher
  - ``pybind11``
  - ``yaml-cpp`` from https://github.com/jbeder/yaml-cpp.git
  - C++ compiler ``nvcc`` and Fortran compiler ``nvfortran``. We recommend installing the `NVIDIA HPC SDK Toolkit <https://developer.nvidia.com/hpc-sdk>`_.
  - ``HDF5``
  - the Python modules specified in ``requirements.txt``

* Additional requirements for building this documentation:

  - Doxygen
  - the Python modules specified in ``docs/requirements.txt``

To build TorchFort, clone the repo and set the environment variable ``NVHPC_CMAKE_DIR`` to the directory where the NVIDIA HPC SDK Toolkit CMake config files are located. Those are usually located in ``<NVHPC root directory>/cmake``. Then, call from the root directory:

.. code-block:: bash

    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=<TorchFort installation prefix> \
          -DNVHPC_CUDA_VERSION=11.8 \
          -DCMAKE_PREFIX_PATH="`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`;${NHPC_CMAKE_DIR}" \
        ..
    make -j install

    
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
