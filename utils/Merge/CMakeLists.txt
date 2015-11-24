cmake_minimum_required(VERSION 3.3)
project(Merge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES merge.cpp)
add_executable(Merge ${SOURCE_FILES})