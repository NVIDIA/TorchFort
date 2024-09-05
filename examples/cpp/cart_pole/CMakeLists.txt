set(cart_pole_example_targets
  train_cart_pole
)

add_library(environments STATIC)
target_sources(environments
  PRIVATE
  env.cpp
)
set_property(TARGET environments PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(train_cart_pole)
target_sources(train_cart_pole
  PRIVATE
  train.cpp
)

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(PyEnvironments py_env.cpp)
target_link_libraries(PyEnvironments PRIVATE environments)

foreach(tgt ${cart_pole_example_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/include
  )
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${tgt} PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${tgt} PRIVATE ${Python_LIBRARIES})
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE ${YAML_CPP_LIBRARY})
  target_link_libraries(${tgt} PRIVATE environments)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  if (TORCHFORT_ENABLE_GPU)
    target_include_directories(${tgt}
      PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)
  endif()
endforeach()

# installation
# executable
install(
  TARGETS ${cart_pole_example_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole
)

# python env
install(
  TARGETS PyEnvironments
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole/python
)

# config files
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_SOURCE_DIR}/config_sim.yaml
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole
)

# python files
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/models.py ${CMAKE_CURRENT_SOURCE_DIR}/python/initialize_models.py ${CMAKE_CURRENT_SOURCE_DIR}/python/visualize.py
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/examples/cpp/cart_pole/python
)
