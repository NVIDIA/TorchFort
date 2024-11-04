include(CMakeFindDependencyMacro)

find_dependency( MPI )
find_package(MPI REQUIRED)

if(TARGET MPI::MPI_C)
message(STATUS "MPI::MPI_C found")
else()
message(STATUS "MPI::MPI_C not found")
endif()

set( CMAKE_PREFIX_PATH "/home/tartarughina/hdf5/build/HDF5-1.17.0-Linux/HDF_Group/HDF5/1.17.0/cmake"; ${CMAKE_PREFIX_PATH})
find_package(HDF5 REQUIRED COMPONENTS Fortran)

if (TARGET HDF5::HDF5)
message(STATUS "HDF5::HDF5 found")
else()
message(STATUS "HDF5::HDF5 not found")
endif()

set(fortran_example_targets
  train
  train_distributed
  train_distributed_um
)

add_executable(train)
target_sources(train
  PRIVATE
  train.f90
  simulation.f90
)

add_executable(train_distributed)
target_sources(train_distributed
  PRIVATE
  train_distributed.f90
  simulation.f90
)

add_executable(train_distributed_um)
target_sources(train_distributed_um
  PRIVATE
  train_distributed_um.cuf
  simulation.f90
)

foreach(tgt ${fortran_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
    ${HDF5_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
  target_link_libraries(${tgt} PRIVATE hdf5::hdf5_fortran)
  target_link_libraries(${tgt} PRIVATE "${PROJECT_NAME}_fort")
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    if(${tgt} STREQUAL "train_distributed_um")
        set(local_CUF_GPU_ARG "${CUF_GPU_ARG},managed")
        target_link_libraries(${tgt} PRIVATE NVHPC::CUDART NVHPC::PGF90RTL NVHPC::ACC)
        target_include_directories(${tgt} PRIVATE ${NVHPC_INCLUDE_DIRS})
    else()
        set(local_CUF_GPU_ARG "${CUF_GPU_ARG}")
    endif()
    set_target_properties(${tgt} PROPERTIES LINKER_LANGUAGE Fortran)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${local_CUF_GPU_ARG}>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-acc -gpu=${local_CUF_GPU_ARG}>)
  elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
  endif()
endforeach()

install(
  TARGETS ${fortran_example_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/simulation
)


install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config_mlp_native.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/config_fcn_torchscript.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_fcn_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/visualize.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/simulation)
