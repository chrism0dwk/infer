set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_SOURCE_DIR}/src/Framework
${CMAKE_SOURCE_DIR}/src/data ${GSL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CMAKE_INCLUDE_PATH})
link_directories(${CMAKE_ARCHIVE_DIR}/src/Framework ${CMAKE_LIBRARY_PATH})

if(OPENMP_FOUND)
add_definitions(-DHAVEOMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(CUDA_FOUND)
include_directories(${CUDA_INCLUDE_DIRS})
include(FindCUDA)

set(CUDA_64_BIT_DEVICE_CODE ON)
set(gpulikelihood_SOURCES GpuLikelihood.cpp GpuLikelihood_gpu.cu)
set(CUDA_NVCC_FLAGS -arch=sm_20)

if(CMAKE_BUILD_TYPE STREQUAL Release)
  set(CUDA_NVCC_FLAGS -use_fast_math;-O3;${CUDA_NVCC_FLAGS})
else()
  set(CUDA_NVCC_FLAGS -g;-G;${CUDA_NVCC_FLAGS})
endif()

cuda_add_library(gpulikelihood STATIC ${gpulikelihood_SOURCES} Likelihood.cpp)
target_link_libraries(gpulikelihood data stlstrtok ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_curand_LIBRARY} cudpp)
endif()

set(cpulikelihood_SOURCES CpuLikelihood.cpp Likelihood.cpp)
add_library(cpulikelihood STATIC ${cpulikelihood_SOURCES})
target_link_libraries(cpulikelihood data stlstrtok)

set(cpuveclikelihood_SOURCES CpuVecLikelihood.cpp Likelihood.cpp)
add_library(cpuveclikelihood STATIC ${cpuveclikelihood_SOURCES})
target_link_libraries(cpuveclikelihood data stlstrtok)

set(mcmc_SOURCES Mcmc.cpp MCMCUpdater.cpp McmcLikelihood.cpp McmcFactory.cpp)
add_library(mcmc STATIC ${mcmc_SOURCES})
add_dependencies(mcmc random)
target_link_libraries(mcmc random)

set(posteriorwriter_SOURCES PosteriorHDF5Writer.cpp PosteriorWriter.cpp)
add_library(posteriorwriter STATIC ${posteriorwriter_SOURCES})
target_link_libraries(posteriorwriter hdf5 hdf5_cpp hdf5_hl hdf5_hl_cpp)

set(posteriorreader_SOURCES PosteriorHDF5Reader.cpp)
add_library(posteriorreader STATIC ${posteriorreader_SOURCES})
target_link_libraries(posteriorreader hdf5 hdf5_cpp hdf5_hl hdf5_hl_cpp)

