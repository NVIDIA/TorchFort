set(fortran_example_targets
  train_graph
)

add_executable(train_graph)
target_sources(train_graph
  PRIVATE
  train.f90
)
set_target_properties(train_graph
                      PROPERTIES OUTPUT_NAME train)

foreach(tgt ${fortran_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${MPI_Fortran_INCLUDE_DIRS}
    ${HDF5_Fortran_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE MPI::MPI_Fortran)
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
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/graph
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_loss.py
        ${CMAKE_CURRENT_SOURCE_DIR}/nodes.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/connectivity.txt
        ${CMAKE_CURRENT_SOURCE_DIR}/visualize.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/fortran/graph)
