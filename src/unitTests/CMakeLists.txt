set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CMAKE_SOURCE_DIR}/src/sim ${CUDA_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIR})
link_directories(${GSL_LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS})

set(fmdMcmc_SOURCES fmdMcmc.cpp)
add_executable(fmdMcmc ${fmdMcmc_SOURCES})
target_link_libraries(fmdMcmc data parameter mcmc gpulikelihood posteriorwriter)

set(fmdSim_SOURCES fmdSim.cpp fmdModel.cpp)
add_executable(fmdSim ${fmdSim_SOURCES})
target_link_libraries(fmdSim data stlstrtok random gsl gslcblas boost_program_options)
