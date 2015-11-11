cmake_minimum_required(VERSION 3.2)
project(PLYConverter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES ply_converter.cpp)
add_executable(ply_converter ${SOURCE_FILES})