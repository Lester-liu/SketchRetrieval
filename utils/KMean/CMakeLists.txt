CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
PROJECT(KMean)

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE(FindCUDA)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_definitions(-D CUDA_ENABLED)

CUDA_ADD_EXECUTABLE(k_mean k_mean_test.cpp k_mean.cpp kernel.cu read_data.cpp)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++0x -ffast-math ")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)

TARGET_LINK_LIBRARIES(k_mean ${OpenCV_LIBS} ${CUDA_CUDART_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})