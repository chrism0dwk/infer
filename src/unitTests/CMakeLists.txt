set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${CMAKE_SOURCE_DIR}/src/mcmc ${CMAKE_SOURCE_DIR}/src/sim ${CUDA_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${CMAKE_INCLUDE_PATH})
link_directories(${GSL_LINK_DIRECTORIES} ${Boost_LIBRARY_DIRS} ${CMAKE_LIBRARY_PATH})

set(theileriaMcmc_SOURCES theileriaMcmc.cpp)
add_executable(theileriaMcmc ${theileriaMcmc_SOURCES})
target_link_libraries(theileriaMcmc data parameter mcmc gpulikelihood posteriorwriter)

set(fmdSim_SOURCES theileriaSim.cpp fmdModel.cpp)
add_executable(theileriaSim ${theileriaSim_SOURCES})
target_link_libraries(theileriaSim data stlstrtok random gsl gslcblas boost_program_options)
