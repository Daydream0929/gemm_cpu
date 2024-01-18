cmake_minimum_required(VERSION 3.12)

project(gemm_cpu LANGUAGES CXX)

# 添加可执行文件
add_executable(a.out main.cpp)

# 添加子目录
add_subdirectory(OpenBLAS)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test!")
add_subdirectory(benchmark)
add_subdirectory(zen3)
add_subdirectory(test)

# 添加链接库
target_link_libraries(a.out zen3_sources)