cmake_minimum_required(VERSION 3.2)
project(Gabor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES gabor.cpp)
add_executable(gabor ${SOURCE_FILES})