///////////////////////////////////////////////////////////////////////
// Name: KernelUtils.cuh					     //
// Created: 2015-07-27						     //
// Author: Chris Jewell <c.jewell@lancaster.ac.uk>		     //
// Copyright: Chris Jewell 2015					     //
// Purpose: CUDA Kernel utilities				     //
///////////////////////////////////////////////////////////////////////


#ifndef KERNELUTILS_CUH
#define KERNELUTILS_CUH

#include <exception>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)

class GpuRuntimeError : public std::exception
{
public:
  GpuRuntimeError(const std::string usrMsg, cudaError_t cudaErr)
  {
    msg_ = "GPU Runtime Error: ";
    msg_ += usrMsg;
    msg_ += " (";
    msg_ += cudaErr;
    msg_ += ",";
    msg_ += cudaGetErrorString(cudaErr);
    msg_ += ")";
  }
  ~GpuRuntimeError() throw ()
  {
  }
  ;
  const char*
  what() const throw ()
  {
    return msg_.c_str();
  }

private:
  std::string msg_;
};


void
__checkCudaError(const cudaError_t err, const char* file, const int line);


template<typename T>
__device__
void
_shmemReduce(T* buff)
{
  // Reduce buffer into output
  __syncthreads();

  for (unsigned int size = blockDim.x / 2; size > 32; size >>= 1)
    {
      if (threadIdx.x < size)
	buff[threadIdx.x] += buff[threadIdx.x + size];
      __syncthreads();
    }
  if (threadIdx.x < 32)
    {
      volatile T* vbuff = buff;
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 32];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 16];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 8];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 4];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 2];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 1];
    }

  __syncthreads();
}

#endif
