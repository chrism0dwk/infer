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
#include <vector>
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


/*  \brief Calculates a distance matrix between 2D coords
 *
 *  Calculates a chunk of a distance matrix between 2D coords in \a coords.
 * 
 *  \param coords the 2D coordinates
 *  \param n the number of coordinates
 *  \param offset the row number at which to start calculating
 *  \param the output buffer to hold the (partial) distance matrix.  Must be 
 *         at least sizeof(float)*n*blockDim.x*gridDim.y.
 *  \param pitch the pitch of the row-major matrix \a buff.
 */
__global__
void
_calcDistance(const float2* coords, const int n, const int offset,
	      float* buff, const size_t pitch)
{
  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = offset + threadIdx.x + blockIdx.y * blockDim.x;

  __shared__ float2 ybuff[DM_THREADSPERBLOCK]; //!< \todo make this dynamic
  
  if(row < n)
    {
      ybuff[threadIdx.x] = coords[row];
    }
  __syncthreads();

  if(col < n)
    {
      float2 x = coords[col];
      int rowlimit = min(blockDim.x, n - offset - blockIdx.y*blockDim.x);

      for(int myrow = 0; myrow < rowlimit; myrow++)
	{
	  float2 y = ybuff[myrow];
	  float dx = x.x - y.x;
	  float dy = x.y - y.y;
	  float d = hypotf(dx, dy);
	  buff[row*pitch + col] = d;
	}
    }
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

/*! \brief Compacts a dense matrix to a CSR network
 *  
 *  Compacts \a n x \a m dense matrix \a dense into a CSR representation 
 *  of a distance-weighted network with edge presence determined by dLimit.
 *
 *  \param dense pointer to row-major dense matrix
 *  \param n number of rows in \a dense
 *  \param m numer of cols in \a dense
 *  \param nnzoffset offset for the rowptr
 *  \param dLimit distance limit
 *  \param csr pointer to the CsrMatrix structure to fill
 */
static
inline
void
compactMatrix(const float* dense, const size_t n, const size_t m, const size_t nnzoffset, const float dLimit, CsrMatrix* csr)
{
  size_t rowptr = 0;
  size_t nnz = nnzoffset;
  csr->rowPtr[0] = nnz;
  for(size_t i=0; i<n; ++i)
    {
      for(size_t j=0; j<m; ++j)
	{
	  float d = dense[i*n + j];
	  if(d < dLimit)
	    {
	      csr->colInd[nnz] = j;
	      csr->val[nnz] = d;
	      ++nnz;
	    }
	}
      ++rowptr;
      csr->rowPtr[rowptr] = nnz;
    }
  csr->nnz = nnz - nnzoffset;
  csr->n = n;
  csr->m = m;
}

/*! \brief Constructs a truncated distance-weighted network
 *
 *  Constructs a network of 2D spatial coordinates, with
 *  edges present if the coordinates are within \a dLimit,
 *  and weighted by the Euclidean distance.
 *
 *  The algorithm works by dividing the rows of the dense distance
 *  matrix into chunks of size \a chunkSize, calculating the distances
 *  on the GPU, and performing row compaction on the CPU.
 *
 *  \param coords an array of float2 with x and y coordinates
 *  \param n the length of \a coords
 *  \param dLimit the distance limit used to define an edge presence
 *  \param chunkSize the number of rows of the matrix per chunk
 *  \return a pointer to an object of type \a CsrMatrix containing a 
 *          compressed row sparse matrix representation of the network.
 */
CsrMatrix*
makeSparseDistance(const float2* coords, const size_t n, const float dLimit, const size_t chunkSize)
{
  // CUDA block dims and streams
  assert(chunkSize % DM_THREADSPERBLOCK == 0);
  dim3 blockDim(DM_THREADSPERBLOCK, 1);
  dim3 gridDim((n + DM_THREADSPERBLOCK - 1)/DM_THREADSPERBLOCK,
	       (chunkSize + DM_THREADSPERBLOCK -1)/DM_THREADSPERBLOCK);
  
  // Allocate memory on the device
  float2* devCoords;
  float* devBuff, *hostBuff;
  size_t buffPitch;
  checkCudaError(cudaMalloc(&devCoords, sizeof(float2)*n));
  checkCudaError(cudaMallocPitch(&devBuff, &buffPitch, sizeof(float)*n, chunkSize));
  hostBuff = new float[n*chunkSize];

  // Copy coords to device
  checkCudaError(cudaMemcpy(devCoords, coords, n, cudaMemcpyHostToDevice));

  // Vectors to hold CSR matrix (will require lots of allocations!)
  std::vector<size_t> rowptr(n+1); rowptr[0] = 0;
  std::vector<size_t> colind;
  std::vector<size_t> value;

  // Loop over row chunks
  CsrMatrix* temp = allocCsrMatrix(chunkSize, n, n*chunkSize, CPU);
  for(size_t offset=0; offset < n; offset += chunkSize)
    {
      _calcDistance<<<gridDim, blockDim>>>(devCoords, n, offset, devBuff, buffPitch);
      checkCudaError(cudaMemcpy2D(hostBuff, n, devBuff, buffPitch, sizeof(float)*n, chunkSize, cudaMemcpyDeviceToHost));
      compactMatrix(hostBuff, chunkSize, n, value.size(), dLimit, temp);

      // Copy data from temp to overall CSR matrix
      std::copy(temp->rowPtr, temp->rowPtr+chunkSize, &rowptr[offset]);
      colind.insert(colind.end(), temp->colInd, temp->colInd+temp->nnz);
      value.insert(value.end(), temp->val, temp->val+temp->nnz);
    }
  rowptr[rowptr.size()-1] = colind.size(); // NNZ

  // Clean up
  destroyCsrMatrix(temp);
  checkCudaError(cudaFree(devCoords));
  checkCudaError(cudaFree(devBuff));
  
  // Package up into CsrMatrix, send to GPU
  CsrMatrix* rv = allocCsrMatrix(n, n, colind.size(), GPU);
  checkCudaError(cudaMemcpy(rv->rowPtr, rowptr.data(), sizeof(size_t)*(n+1), cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(rv->colInd, colind.data(), sizeof(size_t)*colind.size(), cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(rv->val, value.data(), sizeof(float)*value.size(), cudaMemcpyHostToDevice));

  return rv;
}

CsrMatrix*
allocCsrMatrix(const size_t n, const size_t m,
	       const size_t nnz, const DM_PLATFORM platform=CPU)
{
  CsrMatrix* csrMatrix = new CsrMatrix;
  csrMatrix->rowPtr = NULL;
  csrMatrix->colInd = NULL;
  csrMatrix->val = NULL;
  csrMatrix->n = n;
  csrMatrix->m = m;
  csrMatrix->nnz = nnz;
  csrMatrix->platform = platform;

  switch(platform)
    {
    case GPU:
      try {
	checkCudaError(cudaMalloc(&csrMatrix->rowPtr, (n+1)*sizeof(int)));
	checkCudaError(cudaMalloc(&csrMatrix->colInd, nnz*sizeof(int)));
	checkCudaError(cudaMalloc(&csrMatrix->val, nnz*sizeof(float)));
      }
      catch (GpuRuntimeError& e)
	{
	  destroyCsrMatrix(csrMatrix);
	  throw e;
	}
      break;
    case CPU:
      csrMatrix->rowPtr = new int[n+1];
      csrMatrix->colInd = new int[nnz];
      csrMatrix->val    = new float[nnz];
      break;
    }

  return csrMatrix;
}

void
destroyCsrMatrix(CsrMatrix* const& csrMatrix)
{
  switch(csrMatrix->platform)
    {
    case GPU:
      if(csrMatrix->val)
	checkCudaError(cudaFree(csrMatrix->val));
      if(csrMatrix->colInd)
	checkCudaError(cudaFree(csrMatrix->colInd));
      if(csrMatrix->rowPtr)
	checkCudaError(cudaFree(csrMatrix->rowPtr));
      break;
    case CPU:
      if(csrMatrix->val) delete[] csrMatrix->val;
      if(csrMatrix->colInd) delete[] csrMatrix->colInd;
      if(csrMatrix->rowPtr) delete[] csrMatrix->rowPtr;
      break;
    }
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

