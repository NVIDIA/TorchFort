cmake_minimum_required(VERSION 3.14)

set(test_targets
  test_losses
  )

add_executable(test_losses)
target_sources(test_losses
  PRIVATE
  test_losses.cpp
  )

find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)

foreach(tgt ${test_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../
  )
  target_link_libraries(${tgt} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${tgt} PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(${tgt} PRIVATE ${YAML_CPP_LIBRARY})
  target_link_libraries(${tgt} PRIVATE MPI::MPI_CXX)
  target_link_libraries(${tgt} PRIVATE GTest::gtest_main)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  if (TORCHFORT_ENABLE_GPU)
    target_include_directories(${tgt}
      PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)
    target_compile_definitions(${tgt} PRIVATE ENABLE_GPU)
  endif()

  # discover tests: we have an issue with the work dir of gtest so disable that for now
  #gtest_discover_tests(${tgt})
  add_test(NAME ${tgt} COMMAND ${tgt} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# installation
# executable
install(
  TARGETS ${test_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general
)

# copy files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/mse.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/mse_multiarg.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/l1.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/l1_multiarg.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/torchscript.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/torchscript_multiarg.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/torchscript_multiarg_extra.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/torchscript_multiout.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_tests.py DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/general/scripts)
