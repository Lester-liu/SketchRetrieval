CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(CMAKE_FIND_LIBRARY_PREFIXES "lib64;lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so;.a")
SET(CMAKE_SIZEOF_VOID_P "8")

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

SET(VTK_DIR "/usr/local")
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

PROJECT(Sketch)

SET(MAIN src/sketch.cpp)
SET(LEARNING src/learning/svm.cpp src/learning/svm.h src/learning/ml.cpp src/learning/ml.h)
SET(DATA src/structure/blob.cpp src/structure/blob.h)
SET(GEOMETRY src/geometry/triangle.cpp src/geometry/triangle.h src/geometry/line.cpp src/geometry/line.h src/geometry/model.cpp src/geometry/model.h src/geometry/transform.cpp)
ADD_EXECUTABLE(sketch ${MAIN} ${LEARNING} ${DATA} ${GEOMETRY})

TARGET_LINK_LIBRARIES(sketch ${OpenCV_LIBS} ${VTK_LIBRARIES})