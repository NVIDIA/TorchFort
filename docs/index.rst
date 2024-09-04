.. TorchFort documentation master file, created by
   sphinx-quickstart on Wed Jun  1 13:44:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

############################################################################
TorchFort: An Online Deep Learning Interface for HPC programs on NVIDIA GPUs
############################################################################
These pages contain the documentation for TorchFort, an online deep learning interface for HPC programs.

TorchFort is a DL training and inference interface for HPC programs implemented using LibTorch, the C++ backend used by the `PyTorch <https://pytorch.org>`_ framework.
The goal of this library is to help practitioners and domain scientists to seamlessly combine their simulation codes with Deep Learning functionalities available 
within PyTorch.
This library can be invoked directly from Fortran or C/C++ programs, enabling transparent sharing of data arrays to and from the DL framework all contained within the
simulation process (i.e., no external glue/data-sharing code required). The library can directly load PyTorch model definitions exported to TorchScript and implements a
configurable training process that users can control via a simple YAML configuration file format. The configuration files enable users to specify optimizer and loss selection,
learning rate schedules, and much more.

Please contact us or open a GitHub issue if you are interested in using this library
in your own solvers and have questions on usage and/or feature requests.  


Table of Contents
=================
.. toctree::
   :maxdepth: 4

   installation
   usage
   api
   extras


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
