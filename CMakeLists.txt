#CMake top level file for project 'Brandenburg'
project(brandenburg)

cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${brandenburg_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})


### VERSION INFO ###
set(PACKAGE_NAME brandenburg)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 6)

# Get Git version
find_package(Git)

exec_program(${GIT_EXECUTABLE} 
			 ${CMAKE_CURRENT_SOURCE_DIR} 
			 ARGS "log -1 --oneline" 
			 OUTPUT_VARIABLE GITCOMMIT )
			 
string( REGEX REPLACE "^([^ ]+).*" "\\1" VERSION_PATCH "${GITCOMMIT}")

####################



#find_package(CUDA REQUIRED)
find_package(Boost 1.47.0 REQUIRED)
find_package(GSL REQUIRED)

set(LIBS ${LIBS} ${GSL_LIBRARIES} ${GSL_CBLAS_LIBRARIES})

configure_file(${CMAKE_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/config.h @ONLY)


add_subdirectory(src)
