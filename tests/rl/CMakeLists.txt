set(test_targets
  test_replay_buffer
)

add_executable(test_replay_buffer)
target_sources(test_replay_buffer
  PRIVATE
  test_replay_buffer.cpp
)

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
#pybind11_add_module(PyEnvironments py_env.cpp)
#target_link_libraries(PyEnvironments PRIVATE environments)

foreach(tgt ${test_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/include
  )
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${tgt} PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${tgt} PRIVATE ${YAML_CPP_LIBRARY})
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE CUDA::cudart)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
endforeach()

# installation
# executable
install(
  TARGETS ${test_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl
)

# config files
#install(
#  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/config_sim.yaml
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole
#)

# python files
#install(
#  FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/models.py ${CMAKE_CURRENT_SOURCE_DIR}/python/initialize_models.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visualize.py
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole/python
#)
