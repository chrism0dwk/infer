set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${CMAKE_SOURCE_DIR}/src/Framework ${CMAKE_SOURCE_DIR}/src/data ${CUDA_INCLUDE_DIRS})
link_directories(${CMAKE_ARCHIVE_DIR}/src/Framework)

include(FindCUDA)

set(mcmc_SOURCES Mcmc.cpp MCMCUpdater.cpp McmcLikelihood.cpp)
add_library(mcmc STATIC ${mcmc_SOURCES})
add_dependencies(mcmc random)
target_link_libraries(mcmc random)

set(CUDA_64_BIT_DEVICE_CODE ON)
set(gpulikelihood_SOURCES GpuLikelihood.cpp GpuLikelihood_gpu.cu)
set(CUDA_NVCC_FLAGS -arch=sm_12)
cuda_add_library(gpulikelihood STATIC ${gpulikelihood_SOURCES})
target_link_libraries(gpulikelihood data stlstrtok ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
