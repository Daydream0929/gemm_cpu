cmake_minimum_required(VERSION 3.12)
project(test)

# 包括子目录，这里是您的具体测试代码所在的目录
include_directories(${CMAKE_SOURCE_DIR}/zen3/include)

# 添加 zen3 测试目录
add_subdirectory(zen3)