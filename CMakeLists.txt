cmake_minimum_required(VERSION 3.2)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(CMAKE_FIND_LIBRARY_PREFIXES "lib64;lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
SET(CMAKE_SIZEOF_VOID_P "8")

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

project(Sketch)

set(SOURCE_FILES src/main.cpp)
add_executable(sketch ${SOURCE_FILES} src/geometry/triangle.cpp src/geometry/triangle.h src/geometry/line.cpp src/geometry/line.h src/geometry/model.cpp src/geometry/model.h)

TARGET_LINK_LIBRARIES(sketch ${OpenCV_LIBS})