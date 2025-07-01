cmake_minimum_required(VERSION 3.14)

set(test_targets
  test_replay_buffer
  test_rollout_buffer
  test_distributions
  test_off_policy
  test_on_policy
  )

add_executable(test_replay_buffer)
target_sources(test_replay_buffer
  PRIVATE
  test_replay_buffer.cpp
  )

add_executable(test_rollout_buffer)
target_sources(test_rollout_buffer
  PRIVATE
  test_rollout_buffer.cpp
  )

add_executable(test_distributions)
target_sources(test_distributions
  PRIVATE
  test_distributions.cpp
  )

add_executable(test_off_policy)
target_sources(test_off_policy
  PRIVATE
  test_off_policy.cpp
  )

add_executable(test_on_policy)
target_sources(test_on_policy
  PRIVATE
  test_on_policy.cpp
  )


find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)

foreach(tgt ${test_targets})
  target_include_directories(${tgt}
    PRIVATE
    ${YAML_CPP_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/include
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
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl
)

# copy files
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/td3.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/ddpg.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/sac.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl/configs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/configs/ppo.yaml DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl/configs)
