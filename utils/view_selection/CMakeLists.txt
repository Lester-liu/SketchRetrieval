cmake_minimum_required(VERSION 3.2)
project(View_selection)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES view_selection.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(view_selection ${SOURCE_FILES})
target_link_libraries(view_selection ${OpenCV_LIBS})