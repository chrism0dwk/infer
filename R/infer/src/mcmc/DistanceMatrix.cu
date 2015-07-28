///////////////////////////////////////////////////////////////////////
// Name: DistanceMatrix.cpp					     //
// Created: 2015-07-27						     //
// Author: Chris Jewell <c.jewell@lancaster.ac.uk>		     //
// Copyright: Chris Jewell 2015					     //
// Purpose: Implements a truncated distance-weighted network given a //
//          list of 2D coordinates.				     //
///////////////////////////////////////////////////////////////////////

#include <iostream>
#include <stdexcept>
#include <cudpp.h>
#include <cuda_runtime.h>
#include <cassert>

#include "types.hpp"
#include "DistanceMatrix.hpp"
#include "KernelUtils.cuh"



// CUDA Kernels
__global__
void
_numDWithin(const float2* coords, int* output, size_t outputPitch,
	    const float distance, const int n)
{
  // Launch this kernel with block dim [THREADSPERBLOCK,1]
  
  // Get global row/col
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.x + blockIdx.y * blockDim.x;
  float dsq = distance * distance;
  __shared__
    float2 ybuff[DM_THREADSPERBLOCK];
  __shared__
    int nnzbuff[DM_THREADSPERBLOCK];

  nnzbuff[threadIdx.x] = 0.0f;

  if (row < n)
    {
      ybuff[threadIdx.x] = coords[row];
    }
  __syncthreads();

  if (col < n)
    {
      float2 x = coords[col];
      int rowlimit = min(blockDim.x, n - blockIdx.y * blockDim.x);

      for (int myrow = 0; myrow < rowlimit; myrow++)
	{
	  float2 y = ybuff[myrow];
	  float dx = x.x - y.x;
	  float dy = x.y - y.y;
	  float d = dx * dx + dy * dy; //hypotf(dx, dy);
	  nnzbuff[threadIdx.x] += d <= dsq and d > 0.0f;
	}
    }

  _shmemReduce<int>(nnzbuff);
  int* rowptr = (int*) ((char*) output + blockIdx.y * outputPitch);
  rowptr[blockIdx.x] = nnzbuff[0];
}


__global__
void
_computeDrow(const float2* coords, float* devDrow, unsigned int* devIsValid,
	     const int n, const int row, const float distance)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (col < n)
    {
      float dsq = distance * distance;
      float2 y = coords[row];
      float2 x = coords[col];
      float dx = x.x - y.x;
      float dy = x.y - y.y;
      float d = dx * dx + dy * dy; //hypotf(dx,dy);// Require squared distance here!
      devDrow[col] = d;
      devIsValid[col] = (0.0f < d) and (d <= dsq) ? 1 : 0;
    }

}

__global__
void
_fillIndex(int* index, const size_t n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n)
    index[i] = i;
}


// C++ Functions

size_t
numDWithin(const float2* devCoords, const size_t n, const float dLimit)
{
  int* devNNZ;
  size_t devNNZPitch;
  int* hostNNZ;

  // Get Grid dimensions
  dim3 numThreads(DM_THREADSPERBLOCK, 1);
  size_t numBlocks_xy = (n + DM_THREADSPERBLOCK - 1) / DM_THREADSPERBLOCK;
  dim3 numBlocks(numBlocks_xy, numBlocks_xy);

  checkCudaError(
		 cudaMallocPitch(&devNNZ, &devNNZPitch, numBlocks_xy*sizeof(int), numBlocks_xy));
  checkCudaError(
		 cudaMemset2DAsync(devNNZ, devNNZPitch, 0, numBlocks_xy*sizeof(int), numBlocks_xy));

  hostNNZ = new int[numBlocks_xy * numBlocks_xy];

  // Now launch the calculating kernel
  _numDWithin<<<numBlocks,numThreads>>>(devCoords, devNNZ, devNNZPitch, dLimit, n);
  cudaDeviceSynchronize();
  checkCudaError(cudaGetLastError());

  // Copy memory back to host
  checkCudaError(
		 cudaMemcpy2D(hostNNZ, numBlocks_xy*sizeof(int), devNNZ, devNNZPitch, numBlocks_xy*sizeof(int), numBlocks_xy, cudaMemcpyDeviceToHost));

  // Reduction on the host
  int i;
  size_t nnz = 0;
#pragma omp parallel for reduction(+:nnz)
  for (i = 0; i < numBlocks_xy * numBlocks_xy; ++i)
    {
      nnz += hostNNZ[i];
    }

  // Destroy memory
  checkCudaError(cudaFree(devNNZ));
  delete[] hostNNZ;

  return nnz;
}


CsrMatrix*
makeSparseDistance(const float2* coords, const size_t n, const float dLimit)
{
  // Constructs a sparse matrix

  float2* devCoords;
  checkCudaError(cudaMalloc(&devCoords, n*sizeof(float2)));
  checkCudaError(
		 cudaMemcpy(devCoords, coords, n*sizeof(float2), cudaMemcpyHostToDevice));

  // Get number of valid (ie 0 < d <= dLimit) entries
  size_t nnz = numDWithin(devCoords, n, dLimit);
  
  // Allocate the sparse matrix -- may bomb out!
  CsrMatrix* csrMatrix = allocCsrMatrix(n, n, nnz);

  float* devDrow = NULL;
  int* devColNums = NULL;
  unsigned int* devIsValid = NULL;
  try
    {
      cudaMalloc(&devDrow, n * sizeof(float));
      cudaMalloc(&devColNums, n * sizeof(int));
      cudaMalloc(&devIsValid, n * sizeof(unsigned int));
    }
  catch (std::runtime_error& e)
    {
      if (devDrow)
	cudaFree(devDrow);
      if (devColNums)
	cudaFree(devColNums);
      if (devIsValid)
	cudaFree(devIsValid);
      throw e;
    }

  // For each row of the distance matrix
  // 1) Calculate it, outputting valid flags, and indices in output array
  // 2) Compact col and val into respective arrays
  // 3) Enter the rowptr

  // CUDPP bits
  CUDPPHandle theCudpp;
  CUDPPResult result = cudppCreate(&theCudpp);
  if (result != CUDPP_SUCCESS)
    {
      throw std::runtime_error("Could not create the CUDPP instance");
    }

  // Compact plan
  CUDPPConfiguration compactFloatConfig;
  compactFloatConfig.algorithm = CUDPP_COMPACT;
  compactFloatConfig.datatype = CUDPP_FLOAT;
  compactFloatConfig.options = CUDPP_OPTION_FORWARD;
  CUDPPConfiguration compactIntConfig;
  compactIntConfig.algorithm = CUDPP_COMPACT;
  compactIntConfig.datatype = CUDPP_INT;
  compactIntConfig.options = CUDPP_OPTION_FORWARD;
  CUDPPHandle compactFloatPlan;
  result = cudppPlan(theCudpp, &compactFloatPlan, compactFloatConfig, n, 1,
		     0);
  if (result != CUDPP_SUCCESS)
    std::cerr << "Help!  Could not create float plan!" << std::endl;
  CUDPPHandle compactIntPlan;
  result = cudppPlan(theCudpp, &compactIntPlan, compactIntConfig, n, 1, 0);
  if (result != CUDPP_SUCCESS)
    std::cerr << "Help! Could not create int plan!" << std::endl;
  size_t *numValid, *devNumValid;
  checkCudaError(
		 cudaHostAlloc(&numValid, sizeof(size_t), cudaHostAllocMapped));
  checkCudaError(cudaHostGetDevicePointer(&devNumValid, numValid, 0));
  int* hostRowptr = new int[n + 1];
  hostRowptr[0] = 0;

  int numBlocks = (n + DM_THREADSPERBLOCK - 1) / DM_THREADSPERBLOCK;
  _fillIndex<<<numBlocks, DM_THREADSPERBLOCK>>>(devColNums, n);

  for (int row = 0; row < n; ++row)
    {
      // Compute distances, record valid entries
      _computeDrow<<<numBlocks, DM_THREADSPERBLOCK>>>(devCoords, devDrow, devIsValid, n, row, dLimit);
      checkCudaError(cudaGetLastError());

      // Compact into col
      cudppCompact(compactFloatPlan, csrMatrix->val + hostRowptr[row],
		   devNumValid, devDrow, devIsValid, n);
      cudppCompact(compactIntPlan, csrMatrix->colInd + hostRowptr[row],
		   devNumValid, devColNums, devIsValid, n);
      cudaDeviceSynchronize();

      // Update rowptr
      hostRowptr[row + 1] = hostRowptr[row] + (int) *numValid;
    }

  checkCudaError(
		 cudaMemcpy(csrMatrix->rowPtr, hostRowptr, (n+1)*sizeof(int), cudaMemcpyHostToDevice));

  // Clean up
  cudaFree(devDrow);
  cudaFree(devIsValid);
  cudaFree(devColNums);
  cudaFreeHost(numValid);
  delete[] hostRowptr;

  cudppDestroyPlan(compactFloatPlan);
  cudppDestroyPlan(compactIntPlan);
  cudppDestroy(theCudpp);

  cudaFree(devCoords);

  cudaDeviceSynchronize();
  return csrMatrix;
}

CsrMatrix*
allocCsrMatrix(const unsigned long long n, const unsigned long long m, const unsigned long long nnz)
{
  CsrMatrix* csrMatrix = new CsrMatrix;
  csrMatrix->rowPtr = NULL;
  csrMatrix->colInd = NULL;
  csrMatrix->val = NULL;
  csrMatrix->n = n;
  csrMatrix->m = m;
  csrMatrix->nnz = nnz;

  try {
    checkCudaError(cudaMalloc(&csrMatrix->rowPtr, (n+1)*sizeof(size_t)));
    checkCudaError(cudaMalloc(&csrMatrix->colInd, nnz*sizeof(int)));
    checkCudaError(cudaMalloc(&csrMatrix->val, nnz*sizeof(int)));
  }
  catch (GpuRuntimeError& e)
    {
      destroyCsrMatrix(csrMatrix);
      throw e;
    }

  return csrMatrix;
}

void
destroyCsrMatrix(CsrMatrix* const& csrMatrix)
{
  if(csrMatrix->val)
    checkCudaError(cudaFree(csrMatrix->val));
  if(csrMatrix->colInd)
    checkCudaError(cudaFree(csrMatrix->colInd));
  if(csrMatrix->rowPtr)
    checkCudaError(cudaFree(csrMatrix->rowPtr));

  delete csrMatrix;
}


float
getDistElement(const CsrMatrix* d, const int row, const int col) {
  assert(row < d->n);
  assert(col < d->m);
  
  int start = d->rowPtr[row];
  int end = d->rowPtr[row+1];
  for(int j = start; j<end; ++j)
    if (d->colInd[j] == col) return d->val[j];
  return EpiRisk::POSINF;
}

