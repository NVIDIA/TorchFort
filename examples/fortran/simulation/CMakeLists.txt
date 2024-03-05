find_package(HDF5 COMPONENTS Fortran REQUIRED)

# If using nvfortran, build GPU examples which require OpenACC
if (CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  set(fortran_example_gpu_targets
    train
    train_distributed
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

  foreach(tgt ${fortran_example_gpu_targets})
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
    target_compile_definitions(${tgt} PRIVATE ENABLE_OPENACC)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG}>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG}>)
  endforeach()

  install(
    TARGETS ${fortran_example_gpu_targets}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/simulation
  )
endif()

# CPU only examples
set(fortran_example_cpu_targets
  train_cpu
  train_distributed_cpu
)

add_executable(train_cpu)
target_sources(train_cpu
  PRIVATE
  train.f90
  simulation.f90
)

add_executable(train_distributed_cpu)
target_sources(train_distributed_cpu
  PRIVATE
  train_distributed.f90
  simulation.f90
)

foreach(tgt ${fortran_example_cpu_targets})
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
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp>)
  elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -fbackslash>)
  endif()
endforeach()

install(
  TARGETS ${fortran_example_cpu_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/simulation
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config_mlp_native.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/config_fcn_torchscript.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_fcn_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/visualize.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/simulation)
