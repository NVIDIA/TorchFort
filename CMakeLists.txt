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
option(TORCHFORT_BUILD_TESTS "Build tests" OFF)
option(TORCHFORT_ENABLE_GPU "Enable GPU/CUDA support" ON)

# For backward-compatibility with existing variable
if (YAML_CPP_ROOT)
  set(TORCHFORT_YAML_CPP_ROOT ${YAML_CPP_ROOT})
endif()

if (NOT TORCHFORT_YAML_CPP_ROOT)
  message(FATAL_ERROR "Please set TORCHFORT_YAML_CPP_ROOT to yaml-cpp installation directory.")
endif()


if (TORCHFORT_BUILD_FORTRAN)
  set(LANGS Fortran CXX)
else()
  set(LANGS CXX)
endif()

project(torchfort LANGUAGES ${LANGS})

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  # __rdtsc() in torch not supported by nvc++. Use g++ for CXX files.
  message(FATAL_ERROR "TorchFort does not support compilation of C++ files with nvc++. "
                      "Set CMAKE_CXX_COMPILER to g++ to proceed.")
endif()


# unit testing with gtest
if (TORCHFORT_BUILD_TESTS)
  enable_testing()
  include(CTest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

# MPI
find_package(MPI REQUIRED)

# CUDA
if (TORCHFORT_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)

  # HPC SDK
  # Locate and append NVHPC CMake configuration if available
  find_program(NVHPC_CXX_BIN "nvc++")
  if (NVHPC_CXX_BIN)
    string(REPLACE "compilers/bin/nvc++" "cmake" NVHPC_CMAKE_DIR ${NVHPC_CXX_BIN})
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${NVHPC_CMAKE_DIR}")
    find_package(NVHPC COMPONENTS "")
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
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/training.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/losses/l1_loss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/losses/mse_loss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/losses/torchscript_loss.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/cosine_annealing_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/multistep_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/polynomial_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/scheduler_setup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/step_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/lr_schedulers/linear_lr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/models/mlp_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/models/sac_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/models/actor_critic_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/policy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/off_policy/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/off_policy/ddpg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/off_policy/td3.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/off_policy/sac.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/on_policy/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/rl/on_policy/ppo.cpp
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

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
if (TORCHFORT_ENABLE_GPU)
  target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR}
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_GPU)
endif()
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

  if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    # Creating -gpu argument string for nvfortran GPU compilation
    foreach(CUDA_CC ${TORCHFORT_CUDA_CC_LIST})
      list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
    endforeach()
    list(APPEND CUF_GPU_ARG "cuda${NVHPC_CUDA_VERSION}")
    list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)
  endif()


  add_library("${PROJECT_NAME}_fort" SHARED)
  set_target_properties("${PROJECT_NAME}_fort" PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set_target_properties("${PROJECT_NAME}_fort" PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    target_compile_options("${PROJECT_NAME}_fort" PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda>)
  elseif (CMAKE_Fortrain_COMPILER_ID STREQUAL "GNU")
    target_compile_options("${PROJECT_NAME}_fort" PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp>)
  endif()

  # Test for MPI_Comm_f2c/c2f
  try_compile(
    TEST_F2C_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_mpi_f2c.f90
    LINK_LIBRARIES MPI::MPI_Fortran
  )
  if (NOT TEST_F2C_RESULT)
    message(STATUS "Could not link MPI_Comm_f2c in Fortran module. Setting -DMPICH flag during module compilation.")
    target_compile_definitions("${PROJECT_NAME}_fort" PRIVATE MPICH)
  endif()

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
    add_subdirectory(examples/fortran/simulation)
    add_subdirectory(examples/fortran/graph)
  endif()
endif()

# build tests
if (TORCHFORT_BUILD_TESTS)
  add_subdirectory(tests/general)
  add_subdirectory(tests/supervised)
  add_subdirectory(tests/rl)
endif()
