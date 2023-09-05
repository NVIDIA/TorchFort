cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

# https://github.com/NVIDIA/TorchFort/issues/3
cmake_policy(SET CMP0057 NEW)

# __rdtsc() in torch not supported by nvc++. Use g++ for CXX files.
set(CMAKE_CXX_COMPILER "g++")

# User-options
set(TORCHFORT_CUDA_CC_LIST "70;80" CACHE STRING "List of CUDA compute capabilities to build torchfort for.")

project(torchfort CXX Fortran)

# Set TORCH_CUDA_ARCH_LIST string to match TORCHFORT_CUDA_CC_LIST
foreach(CUDA_CC ${TORCHFORT_CUDA_CC_LIST})
    string(REGEX REPLACE "([0-9])$" ".\\1" CUDA_CC_W_DOT ${CUDA_CC})
  list(APPEND TORCH_CUDA_ARCH_LIST ${CUDA_CC_W_DOT})
endforeach()
list(JOIN TORCH_CUDA_ARCH_LIST " " TORCH_CUDA_ARCH_LIST)

# Creating -gpu argument string for Fortran files
foreach(CUDA_CC ${TORCHFORT_CUDA_CC_LIST})
  list(APPEND CUF_GPU_ARG "cc${CUDA_CC}")
endforeach()
list(APPEND CUF_GPU_ARG "cuda${NVHPC_CUDA_VERSION}")
list(JOIN CUF_GPU_ARG "," CUF_GPU_ARG)

find_package(Torch REQUIRED)
find_package(NVHPC REQUIRED COMPONENTS MPI CUDA NCCL)
string(REPLACE "/lib" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})

# Does not work
#find_package(yaml-cpp REQUIRED)

#c/c++ shared library
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

# we want to compile those
target_include_directories(${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE NVHPC::NCCL)
target_link_libraries(${PROJECT_NAME} PRIVATE NVHPC::MPI_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE ${YAML_CPP_ROOT}/lib/libyaml-cpp.a)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${YAML_CPP_ROOT}/include
    ${MPI_CXX_INCLUDE_PATH}
    ${TORCH_INCLUDE_DIRS}
    ${NVHPC_CUDA_INCLUDE_DIR}
    ${NVHPC_NCCL_INCLUDE_DIR}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE YAML_CPP_STATIC_DEFINE)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)

# fortran library and module
add_library("${PROJECT_NAME}_fort" SHARED)
set_target_properties("${PROJECT_NAME}_fort" PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties("${PROJECT_NAME}_fort" PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
target_compile_options("${PROJECT_NAME}_fort" PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -cuda -gpu=${CUF_GPU_ARG}>)


# define public headers
set(public_headers
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort_rl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include/torchfort_enums.h
)
set_target_properties("${PROJECT_NAME}" PROPERTIES PUBLIC_HEADER "${public_headers}")

target_sources("${PROJECT_NAME}_fort"
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fsrc/torchfort_m.F90
)
target_link_libraries(${PROJECT_NAME}_fort MPI::MPI_Fortran)

# install the target and create export-set
install(
  TARGETS ${PROJECT_NAME} "${PROJECT_NAME}_fort"
  EXPORT "${PROJECT_NAME}Targets"
  # except for public headers, as we want them to be inside a library folder
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
  INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)
# install Fortran module
install(FILES ${CMAKE_BINARY_DIR}/include/torchfort.mod DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# install Python files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/python/wandb_helper.py DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/python)

# install docs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs DESTINATION ${CMAKE_INSTALL_PREFIX})

# build examples
add_subdirectory(examples/fortran/simulation)
add_subdirectory(examples/cpp/cart_pole)
