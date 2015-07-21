set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CMAKE_SOURCE_DIR}/src/sim
${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/src/Framework ${GSL_LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS} ${CMAKE_LIBRARY_PATH})

set(testCpuLik_SOURCES testCpuLik.cpp)
add_executable(testCpuLik ${testCpuLik_SOURCES})
target_link_libraries(testCpuLik data parameter mcmc random cpulikelihood)

set(fmdMcmc_SOURCES fmdMcmc.cpp)
add_executable(fmdMcmc ${fmdMcmc_SOURCES})
target_link_libraries(fmdMcmc data parameter mcmc random posteriorwriter cpulikelihood)


if(OPENMP_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(testCpuLik_SOURCES testCpuVecLik.cpp)
add_executable(testCpuVecLik ${testCpuLik_SOURCES})
target_link_libraries(testCpuVecLik data parameter mcmc random cpuveclikelihood)

set(fmdMcmcVec_SOURCES fmdMcmcVec.cpp)
add_executable(fmdMcmcVec ${fmdMcmcVec_SOURCES})
target_link_libraries(fmdMcmcVec data parameter mcmc random posteriorwriter cpuveclikelihood)

endif()

if(CUDA_FOUND)
include_directories(${CUDA_INCLUDE_DIRS})
set(testGpuLik_SOURCES testGpuLik.cpp)
add_executable(testGpuLik ${testGpuLik_SOURCES})
target_link_libraries(testGpuLik data parameter mcmc random gpulikelihood)

set(fmdMcmcGpu_SOURCES fmdMcmcGpu.cpp)
add_executable(fmdMcmcGpu ${fmdMcmcGpu_SOURCES})
target_link_libraries(fmdMcmcGpu data parameter mcmc random posteriorwriter gpulikelihood)

endif()


