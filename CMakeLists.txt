cmake_minimum_required(VERSION 3.12)

project(gemm_cpu LANGUAGES CXX)

# 添加可执行文件
add_executable(a.out main.cpp)

# 添加子目录
add_subdirectory(zen3 benchmark)

# 添加测试目录
add_subdirectory(test)

# 添加链接库
target_link_libraries(a.out zen3_sources)

# # 添加头文件目录
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/zen3/include)