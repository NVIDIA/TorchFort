cmake_minimum_required(VERSION 3.17) # for FindCUDAToolkit support
set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# https://github.com/NVIDIA/TorchFort/issues/3
cmake_policy(SET CMP0057 NEW)

# User-defined build options
set(TORCHFORT_CUDA_CC_LIST "70;80;90" CACHE STRING "List of CUDA compute capabilities to build torchfort for.")
set(TORCHFORT_NCCL_ROOT CACHE STRING "Path to search for NCCL installation. Default NVIDA HPC SDK provided NCCL version if available.")
set(TORCHFORT_YAML_CPP_ROOT CACHE STRING "Path to search for yaml-cpp installation.")
option(TORCHFORT_BUILD_FORTRAN "Build Fortran bindings" ON)
option(TORCHFORT_BUILD_EXAMPLES "Build examples" OFF)

# For backward-compatibility with existing variable
if (YAML_CPP_ROOT)
  set(TORCHFORT_YAML_CPP_ROOT ${YAML_CPP_ROOT})
endif()

if (NOT DEFINED TORCHFORT_YAML_CPP_ROOT)
  message(FATAL_ERROR "Please set TORCHFORT_YAML_CPP_ROOT to yaml-cpp installation directory.")
endif()

# __rdtsc() in torch not supported by nvc++. Use g++ for CXX files.
set(CMAKE_CXX_COMPILER "g++")

if (TORCHFORT_BUILD_FORTRAN)
  set(LANGS CXX Fortran)
else()
  set(LANGS CXX)
endif()
project(torchfort LANGUAGES ${LANGS})

# MPI
find_package(MPI REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# HPC SDK
# Locate and append NVHPC CMake configuration if available
find_program(NVHPC_CXX_BIN "nvc++")
if (NVHPC_CXX_BIN)
  string(REPLACE "compilers/bin/nvc++" "cmake" NVHPC_CMAKE_DIR ${NVHPC_CXX_BIN})
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${NVHPC_CMAKE_DIR}")
  find_package(NVHPC)
endif()
if (NVHPC_FOUND)
  message(STATUS "NVIDIA HPC SDK install located.")
endif()

# Get NCCL library (with optional override)
if (TORCHFORT_NCCL_ROOT)
  find_path(NCCL_INCLUDE_DIR REQUIRED
    NAMES nccl.h
    HINTS ${TORCHFORT_NCCL_ROOT}/include
  )

  find_library(NCCL_LIBRARY REQUIRED
    NAMES nccl
    HINTS ${TORCHFORT_NCCL_ROOT}/lib
  )
else()
  if (NVHPC_FOUND)
    find_package(NVHPC REQUIRED COMPONENTS NCCL)
    find_library(NCCL_LIBRARY
      NAMES nccl
      HINTS ${NVHPC_NCCL_LIBRARY_DIR}
    )
    string(REPLACE "/lib" "/include" NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
  else()
    message(FATAL_ERROR "Cannot find NCCL library. Please set TORCHFORT_NCCL_ROOT to NCCL installation directory.")
  endif()
endif()

message(STATUS "Using NCCL library: ${NCCL_LIBRARY}")

# PyTorch
# Set TORCH_CUDA_ARCH_LIST string to match TORCHFORT_CUDA_CC_LIST
foreach(CUDA_CC ${TORCHFORT_CUDA_CC_LIST})
    string(REGEX REPLACE "([0-9])$" ".\\1" CUDA_CC_W_DOT ${CUDA_CC})
  list(APPEND TORCH_CUDA_ARCH_LIST ${CUDA_CC_W_DOT})
endforeach()
list(JOIN TORCH_CUDA_ARCH_LIST " " TORCH_CUDA_ARCH_LIST)

find_package(Torch REQUIRED)

# yaml-cpp
#find_package(yaml-cpp REQUIRED)
find_path(YAML_CPP_INCLUDE_DIR REQUIRED
  NAMES yaml-cpp/yaml.h
  HINTS ${TORCHFORT_YAML_CPP_ROOT}/include
)
find_library(YAML_CPP_LIBRARY REQUIRED
  NAMES libyaml-cpp.a
  HINTS ${TORCHFORT_YAML_CPP_ROOT}/lib
)
message(STATUS "Using yaml-cpp library: ${YAML_CPP_LIBRARY}")

# C/C++ shared library
add_library(${PROJECT_NAME} SHARED)
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

target_sources(${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/distributed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/logging.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/model_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/model_wrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/model_pack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/param_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/setup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/torchfort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/losses/l1_loss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/losses/mse_loss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/cosine_annealing_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/multistep_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/polynomial_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/scheduler_setup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/step_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/models/mlp_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/rl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/ddpg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/td3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/sac.cpp
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${NCCL_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE ${YAML_CPP_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE YAML_CPP_STATIC_DEFINE)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)

set(public_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort_rl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort_enums.h
)
set_target_properties("${PROJECT_NAME}" PROPERTIES PUBLIC_HEADER "${public_headers}")

install(
  TARGETS ${PROJECT_NAME}
  EXPORT "${PROJECT_NAME}Targets"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# Fortran library and module
if (TORCHFORT_BUILD_FORTRAN)

  if (NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "Fortran support currently requires NVHPC compiler (nvfortran). Set TORCHFORT_BUILD_FORTRAN=0 to disable.")
  endif()

  # Creating -gpu argument string for Fortran files
  foreach(CUDA_CC ${TORCHFORT_CUDA_CC_LIST})
    list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
  endforeach()
  list(APPEND CUF_GPU_ARG "cuda${NVHPC_CUDA_VERSION}")
  list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

  add_library("${PROJECT_NAME}_fort" SHARED)
  set_target_properties("${PROJECT_NAME}_fort" PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set_target_properties("${PROJECT_NAME}_fort" PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
  target_compile_options("${PROJECT_NAME}_fort" PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)

  target_sources("${PROJECT_NAME}_fort"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fsrc/torchfort_m.F90
  )
  target_link_libraries(${PROJECT_NAME}_fort MPI::MPI_Fortran)

  install(
    TARGETS "${PROJECT_NAME}_fort"
  )
  # install Fortran module
  install(FILES ${CMAKE_BINARY_DIR}/include/torchfort.mod DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

# install Python files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/python/wandb_helper.py DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/python)

# install docs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs DESTINATION ${CMAKE_INSTALL_PREFIX})

# build examples
if (TORCHFORT_BUILD_EXAMPLES)
  add_subdirectory(examples/cpp/cart_pole)
  if (TORCHFORT_BUILD_FORTRAN)
    if (NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
      message(FATAL_ERROR "${CMAKE_Fortran_COMPILER_ID} Fortran support currently requires NVHPC compiler (nvfortran). Set TORCHFORT_BUILD_FORTRAN=0 to disable.")
    endif()
    add_subdirectory(examples/fortran/simulation)
  endif()
endif()
