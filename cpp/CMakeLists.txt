cmake_minimum_required(VERSION 3.20)
project(blaze_cpp_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find MKL
find_package(MKL CONFIG REQUIRED)

# Blaze configuration
set(BLAZE_BLAS_MODE ON)
set(BLAZE_BLAS_IS_PARALLEL OFF)  # We want single-threaded for fair comparison

# Include Blaze headers
set(BLAZE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../external/blaze")
if(NOT EXISTS ${BLAZE_INCLUDE_DIR}/blaze/Blaze.h)
    message(FATAL_ERROR "Blaze not found at ${BLAZE_INCLUDE_DIR}. Run 'pixi run setup-blaze' first.")
endif()

# Compiler flags for optimal performance
add_compile_options(-march=native -O3)

# Define BLAZE_BLAS_MODE and use MKL
# MKL uses mkl_cblas.h instead of cblas.h
add_compile_definitions(
    BLAZE_BLAS_MODE=1
    BLAZE_BLAS_IS_PARALLEL=0
    BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION=1
    BLAZE_BLAS_INCLUDE_FILE=<mkl_cblas.h>
)

# Example executable
add_executable(blaze_example example.cpp)
target_include_directories(blaze_example PRIVATE ${BLAZE_INCLUDE_DIR})
target_compile_options(blaze_example PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(blaze_example PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(blaze_example PRIVATE MKL::MKL)

# Benchmark executable
add_executable(blaze_benchmark benchmark.cpp)
target_include_directories(blaze_benchmark PRIVATE ${BLAZE_INCLUDE_DIR})
target_compile_options(blaze_benchmark PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(blaze_benchmark PRIVATE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(blaze_benchmark PRIVATE MKL::MKL)
