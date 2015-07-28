///////////////////////////////////////////////////////////////////////
// Name: KernelUtils.cu 					     //
// Created: 2015-07-27						     //
// Author: Chris Jewell <c.jewell@lancaster.ac.uk>		     //
// Copyright: Chris Jewell 2015					     //
// Purpose: CUDA Kernel utilities				     //
///////////////////////////////////////////////////////////////////////

#include "KernelUtils.cuh"


void
__checkCudaError(const cudaError_t err, const char* file, const int line)
{
  if (err != cudaSuccess)
    {
      std::stringstream s;
      s << file << "(" << line << ") : Cuda Runtime error ";
      throw GpuRuntimeError(s.str(), err);
    }
}

