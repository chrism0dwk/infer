set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CMAKE_SOURCE_DIR}/src/sim ${CUDA_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/src/Framework ${GSL_LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS} ${CMAKE_LIBRARY_PATH})

set(fmdMcmc_SOURCES fmdMcmc.cpp)
add_executable(fmdMcmc ${fmdMcmc_SOURCES})
target_link_libraries(fmdMcmc data parameter mcmc gpulikelihood posteriorwriter)

set(fmdSim_SOURCES fmdSim.cpp fmdModel.cpp)
add_executable(fmdSim ${fmdSim_SOURCES})
target_link_libraries(fmdSim data stlstrtok random gsl gslcblas boost_program_options)

set(testCpuLik_SOURCES testCpuLik.cpp)
add_executable(testCpuLik ${testCpuLik_SOURCES})
set_target_properties(testCpuLik PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(testCpuLik data parameter random cpulikelihood)

set(testGpuLik_SOURCES testGpuLik.cpp)
add_executable(testGpuLik ${testGpuLik_SOURCES})
target_link_libraries(testGpuLik data parameter random gpulikelihood)
