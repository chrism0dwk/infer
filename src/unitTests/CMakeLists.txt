set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CMAKE_SOURCE_DIR}/src/sim ${CUDA_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/src/Framework ${GSL_LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS} ${CMAKE_LIBRARY_PATH})



set(fmdSim_SOURCES fmdSim.cpp fmdModel.cpp)
add_executable(fmdSim ${fmdSim_SOURCES})
target_link_libraries(fmdSim data stlstrtok random gsl gslcblas boost_program_options)




set(testCpuLik_SOURCES testCpuLik.cpp)
add_executable(testCpuLik ${testCpuLik_SOURCES})
set_target_properties(testCpuLik PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(testCpuLik data parameter mcmc random cpulikelihood)

set(testCpuLik_SOURCES testCpuVecLik.cpp)
add_executable(testCpuVecLik ${testCpuLik_SOURCES})
set_target_properties(testCpuVecLik PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(testCpuVecLik data parameter mcmc random cpuveclikelihood)

set(testGpuLik_SOURCES testGpuLik.cpp)
add_executable(testGpuLik ${testGpuLik_SOURCES})
target_link_libraries(testGpuLik data parameter mcmc random gpulikelihood)

set(testBoth_SOURCES testBoth.cpp)
add_executable(testBoth ${testBoth_SOURCES})
set_target_properties(testBoth PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(testBoth data parameter random gpulikelihood cpulikelihood)


set(fmdMcmc_SOURCES fmdMcmc.cpp)
add_executable(fmdMcmc ${fmdMcmc_SOURCES})
set_target_properties(fmdMcmc PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(fmdMcmc data parameter mcmc cpulikelihood posteriorwriter)

set(fmdMcmcVec_SOURCES fmdMcmcVec.cpp)
add_executable(fmdMcmcVec ${fmdMcmcVec_SOURCES})
set_target_properties(fmdMcmcVec PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(fmdMcmcVec data parameter mcmc cpuveclikelihood posteriorwriter)

set(fmdMcmcGpu_SOURCES fmdMcmcGpu.cpp)
add_executable(fmdMcmcGpu ${fmdMcmcGpu_SOURCES})
set_target_properties(fmdMcmcGpu PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(fmdMcmcGpu data parameter mcmc gpulikelihood posteriorwriter)

set(fmdMcmcDummy_SOURCES fmdMcmcDummy.cpp)
add_executable(fmdMcmcDummy ${fmdMcmcDummy_SOURCES})
set_target_properties(fmdMcmcDummy PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
target_link_libraries(fmdMcmcDummy data parameter mcmc dummylikelihood posteriorwriter)
