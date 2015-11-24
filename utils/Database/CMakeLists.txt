cmake_minimum_required(VERSION 3.2)
project(Database)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES database.cpp)
add_executable(build_database ${SOURCE_FILES})