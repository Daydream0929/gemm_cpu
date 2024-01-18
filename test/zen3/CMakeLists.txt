cmake_minimum_required(VERSION 3.12)
project(zen3_test)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 查找 Google Benchmark
find_package(benchmark REQUIRED)

# 查找 OpenBLAS
find_package(OpenBLAS REQUIRED)

# 添加测试可执行文件
add_executable(sgemm_test sgemm_test.cc)

# 链接 Google Benchmark 和其他必要的库
target_link_libraries(sgemm_test benchmark::benchmark zen3_sources OpenBLAS::OpenBLAS) 

