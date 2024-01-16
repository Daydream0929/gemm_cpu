cmake_minimum_required(VERSION 3.12)
project(gemm_cpu LANGUAGES CXX)

add_subdirectory(zen3)

add_executable(a.out main.cpp)

target_link_libraries(a.out PUBLIC zen3)