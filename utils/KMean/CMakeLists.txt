cmake_minimum_required(VERSION 3.2)
project(KMean)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES k_mean.cpp)
add_executable(k_mean ${SOURCE_FILES})