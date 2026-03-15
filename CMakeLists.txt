cmake_minimum_required(VERSION 3.16)
project(minitensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(minitf src/main.cpp src/Tensor.cpp src/Node.cpp src/Ops.cpp src/Engine.cpp)
target_include_directories(minitf PRIVATE include)
target_compile_options(minitf PRIVATE -O2 -g -Wall -Wextra -Werror)
