cmake_minimum_required(VERSION 3.20)
project(chunk_scan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require CUDA
find_package(CUDAToolkit REQUIRED)

# Target sm_80+ (Ampere and above) for tensor core and warp-level intrinsics
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(src)

# Collect all CUDA source files
file(GLOB_RECURSE SRC_FILES src/*.cu)

# Static library containing all kernel implementations
add_library(chunk_scan_lib STATIC ${SRC_FILES})
target_include_directories(chunk_scan_lib PUBLIC src)
set_target_properties(chunk_scan_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Correctness tests
file(GLOB_RECURSE TEST_FILES tests/*.cu)
if(TEST_FILES)
    add_executable(test_correctness ${TEST_FILES})
    target_link_libraries(test_correctness chunk_scan_lib CUDA::cudart)
    set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Latency benchmarks
file(GLOB_RECURSE BENCH_FILES benchmarks/*.cu)
if(BENCH_FILES)
    add_executable(bench_latency ${BENCH_FILES})
    target_link_libraries(bench_latency chunk_scan_lib CUDA::cudart)
    set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
