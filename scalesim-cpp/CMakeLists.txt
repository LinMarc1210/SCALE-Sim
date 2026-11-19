cmake_minimum_required(VERSION 3.16)
project(scalesim_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Bit-exact parity with the Python reference: forbid FMA contraction and fast-math so every
# floating point expression is evaluated exactly like CPython / numpy do.
add_compile_options(-Wall -Wextra -ffp-contract=off -fno-fast-math)

find_package(Threads REQUIRED)

add_executable(scalesim_cpp
  src/main.cpp
  src/config.cpp
  src/topology.cpp
  src/memory.cpp
  src/layer_sim.cpp)
target_link_libraries(scalesim_cpp PRIVATE Threads::Threads)

# Helper used by verify/check_pyfmt_pyset.py to validate float formatting and set ordering.
add_executable(scalesim_selftest verify/selftest.cpp)
target_include_directories(scalesim_selftest PRIVATE src)
