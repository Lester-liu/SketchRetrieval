cmake_minimum_required(VERSION 3.2)
project(Sample)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES sample.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(sample ${SOURCE_FILES})
target_link_libraries(sample ${OpenCV_LIBS})