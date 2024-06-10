cmake_minimum_required(VERSION 3.14)

# GoogleTest requires at least C++14
#set(CMAKE_CXX_STANDARD 17)
#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#FetchContent_MakeAvailable(googletest)

# enable test discovery
#enable_testing()
#include(GoogleTest)

set(test_targets
  test_replay_buffer
  test_rollout_buffer
)

add_executable(test_replay_buffer)
target_sources(test_replay_buffer
  PRIVATE
  test_replay_buffer.cpp
  )
#add_test(NAME TestReplayBuffer COMMAND test_replay_buffer)

add_executable(test_rollout_buffer)
target_sources(test_rollout_buffer
  PRIVATE
  test_rollout_buffer.cpp
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
  target_link_libraries(${tgt} PRIVATE GTest::gtest_main)
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)
  target_link_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>)

  # discover tests
  gtest_discover_tests(${tgt})
endforeach()

# installation
# executable
install(
  TARGETS ${test_targets}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/tests/rl
)
