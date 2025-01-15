find_package(HDF5 COMPONENTS Fortran REQUIRED)

set(fortran_example_targets
  train
  train_distributed
)

add_executable(train)
target_sources(train
  PRIVATE
  train.f90
  simulation.f90
)
set_target_properties(train PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/0 )

add_executable(train_distributed)
target_sources(train_distributed
  PRIVATE
  train_distributed.f90
  simulation.f90
)
set_target_properties(train_distributed PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mod/1 )

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
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cpp -acc -gpu=${CUF_GPU_ARG}>)
    target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>: -acc -gpu=${CUF_GPU_ARG}>)
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
