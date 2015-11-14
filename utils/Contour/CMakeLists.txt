cmake_minimum_required(VERSION 3.3)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)

project(Contour)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES canny.cpp)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
add_executable(contour ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(contour ${OpenCV_LIBS})