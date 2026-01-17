cmake_minimum_required(VERSION 3.20)
project(blaze_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Blaze header-only library
set(BLAZE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../blaze")

# MKL configuration
set(MKL_ROOT "$ENV{MKLROOT}")

# Configure Blaze for MKL BLAS
add_definitions(-DBLAZE_BLAS_MODE=1)
add_definitions(-DBLAZE_BLAS_IS_64BIT=0)
add_definitions(-DBLAZE_BLAS_IS_PARALLEL=0)
add_definitions(-DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0)
# Use MKL's CBLAS header
add_definitions(-DBLAZE_BLAS_INCLUDE_FILE=<mkl_cblas.h>)

# Create benchmark executable
add_executable(blaze_bench main.cpp)

target_include_directories(blaze_bench PRIVATE 
    ${BLAZE_INCLUDE_DIR}
    ${MKL_ROOT}/include
)

# Link MKL (sequential, lp64)
target_link_directories(blaze_bench PRIVATE ${MKL_ROOT}/lib)
target_link_libraries(blaze_bench PRIVATE 
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    pthread
    m
    dl
)

# Create example executable
add_executable(blaze_example example.cpp)

target_include_directories(blaze_example PRIVATE 
    ${BLAZE_INCLUDE_DIR}
    ${MKL_ROOT}/include
)

target_link_directories(blaze_example PRIVATE ${MKL_ROOT}/lib)
target_link_libraries(blaze_example PRIVATE 
    mkl_intel_lp64
    mkl_sequential
    mkl_core
    pthread
    m
    dl
)
