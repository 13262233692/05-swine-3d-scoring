cmake_minimum_required(VERSION 3.18)
project(swine_3d_scoring VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(WITH_TENSORRT "Enable TensorRT acceleration" ON)
option(WITH_REALSENSE "Enable Intel RealSense SDK" ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
if(EXISTS "${ONNXRUNTIME_ROOT}/include")
    include_directories("${ONNXRUNTIME_ROOT}/include")
    link_directories("${ONNXRUNTIME_ROOT}/lib")
endif()

if(WITH_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT root directory")
    if(EXISTS "${TENSORRT_ROOT}/include")
        include_directories("${TENSORRT_ROOT}/include")
        link_directories("${TENSORRT_ROOT}/lib")
    endif()
endif()

if(WITH_REALSENSE)
    find_package(realsense2 QUIET)
    if(realsense2_FOUND)
        include_directories(${realsense2_INCLUDE_DIR})
        add_definitions(-DUSE_REALSENSE)
    endif()
endif()

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.cu"
)

file(GLOB_RECURSE HEADERS
    "include/*.h"
    "include/*.hpp"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    CUDA::cudart
)

if(EXISTS "${ONNXRUNTIME_ROOT}/lib")
    if(WIN32)
        target_link_libraries(${PROJECT_NAME} onnxruntime)
    else()
        target_link_libraries(${PROJECT_NAME} libonnxruntime.so)
    endif()
endif()

if(WITH_TENSORRT AND EXISTS "${TENSORRT_ROOT}/lib")
    target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser)
endif()

if(realsense2_FOUND)
    target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})
endif()
