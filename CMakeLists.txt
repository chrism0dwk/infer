#CMake top level file for project 'Brandenburg'
project(brandenburg)

set(PACKAGE_NAME brandenburg)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 6)
set(VERSION_PATCH 10)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(Boost 1.47.0 REQUIRED)
find_package(GSL REQUIRED)

set(CMAKE_MODULE_PATH ${brandenburg_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h @ONLY)


add_subdirectory(src)
