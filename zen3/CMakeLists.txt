cmake_minimum_required(VERSION 3.12)

project(zen3 LANGUAGES CXX)

# 添加源文件
set(SOURCES 
    src/gemm.cc    
)

# 添加头目录文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 生成静态库
add_library(zen3_sources ${SOURCES})

# 将头文件添加到库的公共接口
target_include_directories(zen3_sources PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

