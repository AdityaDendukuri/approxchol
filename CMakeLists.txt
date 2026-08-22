cmake_minimum_required(VERSION 3.15)
project(approxchol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(approxchol INTERFACE)

target_include_directories(approxchol
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

enable_testing()

add_executable(test_approxchol tests/test_approxchol.cpp)
target_link_libraries(test_approxchol PRIVATE approxchol)

add_test(NAME test_approxchol COMMAND test_approxchol)

add_executable(benchmark_cpp benchmarks/benchmark_cpp.cpp)
target_link_libraries(benchmark_cpp PRIVATE approxchol)
target_include_directories(benchmark_cpp PRIVATE benchmarks)
