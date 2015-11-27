cmake_minimum_required(VERSION 3.2)
project(Merge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES merge.cpp)
add_executable(merge ${SOURCE_FILES})