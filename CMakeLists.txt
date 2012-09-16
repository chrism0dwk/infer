#CMake top level file for project 'InFER'
project(infer-fmd)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${infer-fmd_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

set(CMAKE_VERBOSE_MAKEFILE true)
### VERSION INFO ###
set(PACKAGE_NAME infer-fmd)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 7)

# Get Git version
find_package(Git)

execute_process(COMMAND ${GIT_EXECUTABLE} 
			 ${CMAKE_CURRENT_SOURCE_DIR} 
			 ARGS "log -1 --oneline" 
			 OUTPUT_VARIABLE GITCOMMIT )
			 
string( REGEX REPLACE "^([^ ]+).*" "\\1" VERSION_PATCH "${GITCOMMIT}")

####################



find_package(CUDA REQUIRED)
find_package(Boost 1.47.0 REQUIRED)
find_package(GSL REQUIRED)

set(LIBS ${LIBS} ${GSL_LIBRARIES} ${GSL_CBLAS_LIBRARIES})

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h @ONLY)


add_subdirectory(src)
