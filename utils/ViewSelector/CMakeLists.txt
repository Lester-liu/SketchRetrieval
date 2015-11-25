cmake_minimum_required(VERSION 3.2)
project(View_selection)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES selector.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(selector ${SOURCE_FILES})
target_link_libraries(selector ${OpenCV_LIBS})