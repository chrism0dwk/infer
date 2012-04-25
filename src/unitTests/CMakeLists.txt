set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework ${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CUDA_INCLUDE_DIRS})

set(fmdMcmc_SOURCES fmdMcmc.cpp)
add_executable(fmdMcmc ${fmdMcmc_SOURCES})
target_link_libraries(fmdMcmc data parameter mcmc gpulikelihood)
