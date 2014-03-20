/*
 * Gpulikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <math_functions.h>
#include <device_functions.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <gsl/gsl_cdf.h>

#include <assert.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

//#define ALPHA 0.3

#include "GpuLikelihood.hpp"

namespace EpiRisk
{
  // Constants
  const float UNITY = 1.0;
  const float ZERO = 0.0;
#define PI (atanf(1.0f)*4.0f)
  
  float
  GetDistElement(const CsrMatrix* d, const int row, const int col) {
    assert(row < d->n);
    assert(col < d->m);
    
    int start = d->rowPtr[row];
    int end = d->rowPtr[row+1];
    for(int j = start; j<end; ++j)
      if (d->colInd[j] == col) return d->val[j];
    return EpiRisk::POSINF;
  }


  inline
  float
  timeinseconds(const timeval a, const timeval b)
  {
    timeval result;
    timersub(&b, &a, &result);
    return result.tv_sec + result.tv_usec / 1000000.0;
  }

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
  __checkCudaError(const cudaError_t err, const char* file, const int line)
  {
    if (err != cudaSuccess)
      {
        std::stringstream s;
        s << file << "(" << line << ") : Cuda Runtime error ";
        throw GpuRuntimeError(s.str(), err);
      }
  }

  // CUDA kernels

  __device__
  float cache[5];

  template<typename T>
  struct Log
  {
    __host__ __device__
    T
    operator()(const T& val) const
    {
      return logf(val);
    }
  };

  template<typename T>
  struct LessThanZero
  {
    __host__ __device__
    bool
    operator()(const T& val) const
    {
      return val < 0;
    }
  };


  template<typename T>
  struct IndirectMin
  {
    __host__ __device__
    IndirectMin(T* ptr) :
      ptr_(ptr)
    {
    }
    ;

    __host__ __device__
    bool
    operator()(const InfecIdx_t& lhs, const InfecIdx_t& rhs) const
    {
      return ptr_[lhs.ptr] < ptr_[rhs.ptr];
    }
  private:
    T* ptr_;
  };

 __device__ float
_h(const float t, const float I, float nu, float ys, float yw)
{
  // Periodic piece-wise cubic spline
  float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  float Y[] = {1.0f, 1.0f, 1.0f, 1.0f,1.0f};
  float delta = 0.25;
  
  assert(t-I >= 0);
  
  // Re-scale time to unit period
  float tAdj = (t+nu)/365.0f;
  tAdj = tAdj - floorf(tAdj);

  // Set up parameters
  Y[0] = ys; Y[2] = yw; Y[4] = ys;
  
  // Calculate spline value
  int epoch = (int)(tAdj*4.0f);

  float a = -6.0f*(Y[epoch+1]-Y[epoch])/(delta*delta);
  float b = -a;
  
  float h = a/(6.0f*delta) * powf(tAdj - T[epoch], 3);
  h      += (Y[epoch+1]/delta - (a*delta)/6.0f) *  (tAdj - T[epoch]);
  h      += b/(6.0f*delta) * powf(T[epoch+1] - tAdj, 3);
  h      += (Y[epoch]/delta - (b*delta)/6.0f) * (T[epoch+1] - tAdj);
  
  return h;
}
  
  __device__ float
  _HIntegrand(float t, const float* T, const float* Y) {
    // Calculates cubic spline integral between t1 and t2
    float delta = 0.25f;
    float a = -6.0f * (Y[1] - Y[0])/(delta * delta);
    float b = -a;
    
    float h;
    h =  a/(24.0f*delta) * powf(t - T[0], 4);
    h += (Y[1]/(2.0f*delta) - (a*delta)/12.0f) * powf(t - T[0],2);
    h -= b/(24.0f*delta) * powf(T[1] - t, 4);
    h -= (Y[0]/(2.0f*delta) - (b*delta)/12.0f) * powf(T[1] - t,2);
      
    return h;
  }

__global__ void
_HIntegConst(const float ys, const float yw, float* cache)
{
  // Calculates cached integral -- requires only 4 threads
  float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  float Y[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  
  __shared__ float buff[4];
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  Y[0] = ys; Y[2] = yw; Y[4] = ys;
  
  if(tid < 4) {
    buff[tid] = _HIntegrand(T[tid+1], T+tid, Y+tid)
    - _HIntegrand(T[tid], T+tid, Y+tid);
    
    __syncthreads();
    
    // Reduce cummulative sum here -- needs parallelising
    if(tid == 0) {
      buff[1] += buff[0];
      buff[2] += buff[1];
      buff[3] += buff[2];
    }
    __syncthreads();
    
    cache[tid+1] = buff[tid];  // Cache[0] is set to 0
  }
}
  
  void
  CalcHFuncIntegCache(const float ys, const float yw, float* cache, const bool setZero=false)
  {
    // Calculates the CUDA H function integral cache
    
    if(setZero)
      checkCudaError(cudaMemset(cache, 0, sizeof(float)));
    
    _HIntegConst<<<1, 4>>>(ys, yw, cache);

    cudaDeviceSynchronize();
  }
  
__device__ float
_H(const float b, const float a, const float nu, const float alpha1, const float alpha2, const float* hCache)
{
  // Returns the integral of the 'h' function over [a,b]
  float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  float Y[] = {1.0f, 1.0f,  1.0f, 1.0f,  1.0f};
  float delta = 0.25f;
  
  Y[0] = alpha1; Y[2] = alpha2; Y[4] = alpha1;
  
  if(b <= a) return 0.0f;
  
  float t1 = (a+nu)/365.0f;
  float t2 = (b+nu)/365.0f;
  
  // Set relative to the beginning of t1's period
  t2 = t2 - floorf(t1);
  t1 = t1 - floorf(t1);
  
  int epoch1 = t1*4;
  int period2 = t2;
  int epoch2 = (t2-floorf(t2))*4;

  float integrand1 = hCache[epoch1] + _HIntegrand(t1, T+epoch1, Y+epoch1) - _HIntegrand(epoch1*0.25f, T+epoch1, Y+epoch1);
  float integrand2 = hCache[4]*period2 + hCache[epoch2] + _HIntegrand(t2-period2, T+epoch2, Y+epoch2) - _HIntegrand(epoch2*0.25f, T+epoch2, Y+epoch2);

  return 365.0f*(integrand2 - integrand1);
  
}

  struct DistanceKernel {
    __device__ __host__ float
    operator()(const float dsq, const float delta, const float omega)
    {
      return delta / powf(delta*delta + dsq, omega);
    }
  };
    
  struct Identity {
    __device__ __host__ float
    operator()(const float d, const float delta, const float omega)
    {
      return d;
    }
  };

  __device__ float
  _atomicAdd(float* address, float val)
  {
    unsigned int* address_as_ui = (unsigned int*) address;
    unsigned int old = *address_as_ui, assumed;
    do
      {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed,
			__float_as_int(val + __int_as_float(assumed)));
      }
    while (assumed != old);
    return __int_as_float(old);
  }

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

  __global__
  void
  _computeDistance(const float2* coords, int* output, size_t outputPitch,
		   const float distance, const int n)
  {
    // Launch this kernel with block dim [THREADSPERBLOCK,1]

    // Get global row/col
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.x + blockIdx.y * blockDim.x;
    float dsq = distance * distance;
    __shared__
      float2 ybuff[THREADSPERBLOCK];
    __shared__
      int nnzbuff[THREADSPERBLOCK];

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

  size_t
  distanceNumValid(const float2* devCoords, const size_t n, const float dLimit)
  {
    int* devNNZ;
    size_t devNNZPitch;
    int* hostNNZ;

    // Get Grid dimensions
    dim3 numThreads(THREADSPERBLOCK, 1);
    size_t numBlocks_xy = (n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    dim3 numBlocks(numBlocks_xy, numBlocks_xy);

    checkCudaError(
		   cudaMallocPitch(&devNNZ, &devNNZPitch, numBlocks_xy*sizeof(int), numBlocks_xy));
    checkCudaError(
		   cudaMemset2DAsync(devNNZ, devNNZPitch, 0, numBlocks_xy*sizeof(int), numBlocks_xy));

    hostNNZ = new int[numBlocks_xy * numBlocks_xy];

    // Now launch the calculating kernel
    _computeDistance<<<numBlocks,numThreads>>>(devCoords, devNNZ, devNNZPitch, dLimit, n);
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
    size_t nnz = distanceNumValid(devCoords, n, dLimit);
    // Allocate the sparse matrix -- may bomb out!
    CsrMatrix* csrMatrix = new CsrMatrix;
    csrMatrix->rowPtr = NULL;
    csrMatrix->colInd = NULL;
    csrMatrix->val = NULL;
    csrMatrix->valtr = NULL;
    csrMatrix->nnz = nnz;
    csrMatrix->m = n;
    csrMatrix->n = n;

    float* devDrow = NULL;
    int* devColNums = NULL;
    unsigned int* devIsValid = NULL;

    try
      {
        cudaMalloc(&csrMatrix->rowPtr, (n + 1) * sizeof(size_t));
        cudaMalloc(&csrMatrix->colInd, nnz * sizeof(int));
        cudaMalloc(&csrMatrix->val, nnz * sizeof(int));
	csrMatrix->valtr = csrMatrix->val;
        cudaMalloc(&devDrow, n * sizeof(float));
        cudaMalloc(&devColNums, n * sizeof(int));
        cudaMalloc(&devIsValid, n * sizeof(unsigned int));
      }
    catch (runtime_error& e)
      {
        if (csrMatrix->rowPtr)
          cudaFree(csrMatrix->rowPtr);
        if (csrMatrix->colInd)
          cudaFree(csrMatrix->colInd);
        if (csrMatrix->val) {
          cudaFree(csrMatrix->val);
	  csrMatrix->valtr = NULL;
	}
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
        throw runtime_error("Could not create the CUDPP instance");
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
      cerr << "Help!  Could not create float plan!" << endl;
    CUDPPHandle compactIntPlan;
    result = cudppPlan(theCudpp, &compactIntPlan, compactIntConfig, n, 1, 0);
    if (result != CUDPP_SUCCESS)
      cerr << "Help! Could not create int plan!" << endl;
    size_t *numValid, *devNumValid;
    checkCudaError(
		   cudaHostAlloc(&numValid, sizeof(size_t), cudaHostAllocMapped));
    checkCudaError(cudaHostGetDevicePointer(&devNumValid, numValid, 0));
    int* hostRowptr = new int[n + 1];
    hostRowptr[0] = 0;

    int numBlocks = (n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    _fillIndex<<<numBlocks, THREADSPERBLOCK>>>(devColNums, n);

    for (int row = 0; row < n; ++row)
      {
        // Compute distances, record valid entries
	_computeDrow<<<numBlocks, THREADSPERBLOCK>>>(devCoords, devDrow, devIsValid, n, row, dLimit);
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

  void
  destroyCsrMatrix(CsrMatrix* csrMatrix)
  {
    if(csrMatrix->val == csrMatrix->valtr) {
      checkCudaError(cudaFree(csrMatrix->val));
      csrMatrix->valtr = NULL;
    }
    else {
      checkCudaError(cudaFree(csrMatrix->val));
      checkCudaError(cudaFree(csrMatrix->valtr));
    }
    checkCudaError(cudaFree(csrMatrix->colInd));
    checkCudaError(cudaFree(csrMatrix->rowPtr));

    delete csrMatrix;
  }


  bool
  getDistMatrixElement(const int row, const int col, const CsrMatrix* csrMatrix, float* val)
  {
    int* cols = csrMatrix->colInd + csrMatrix->rowPtr[row];
    float* vals = csrMatrix->val + csrMatrix->rowPtr[row];
    int rowlen = csrMatrix->rowPtr[row+1] - csrMatrix->rowPtr[row];

    for(int ptr=0; ptr<rowlen; ++ptr)
      {
        if(cols[ptr] == col) {
          *val = vals[ptr];
          return true;
        }
      }
    return false;
  }


  int
  checkDistMatrixSymmetry(const CsrMatrix* csrMatrix)
  {

    int row;
    int nonsymmetric = 0;
#pragma omp parallel for shared(csrMatrix) private(row) reduction(+:nonsymmetric)
    for(row=0; row<csrMatrix->n; ++row)
      {
        int rowptr = csrMatrix->rowPtr[row];
        int cRowLen = csrMatrix->rowPtr[row+1] - rowptr;

        for(size_t colidx=0; colidx<cRowLen; ++colidx)
          {
            int colnum = csrMatrix->colInd[rowptr + colidx];
            float rtoc = csrMatrix->val[rowptr + colidx];
            float ctor;
            bool rv = getDistMatrixElement(colnum,row,csrMatrix,&ctor);

	    if(rtoc != ctor or !rv) {
#pragma omp critical
	      {
		cerr << "Non-symmetry: (" << row << "," << colnum << ") = " << rtoc
		     << " but (" << colnum << "," << row << ") = " << ctor;
		cerr << endl;
	      }
	      nonsymmetric += 1;
	    }
          }
      }

    return nonsymmetric;
  }


  __global__ void
  _sanitizeEventTimes(float* data, int pitch, const float time, const int size)
  {
    // Ensures Ii <= Ni <= Ri for individual i
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < size)
      {
        float R = data[tid + pitch * 2];
        float N = data[tid + pitch];
        float I = data[tid];

        R = fminf(R, time);
        N = fminf(N, R);
        I = fminf(I, N);

        data[tid + pitch * 2] = R;
        data[tid + pitch] = N;
        data[tid] = I;
      }
  }

  __global__ void
  _calcSusceptibility(const int* tickLevel, const int popSize, const float* phi, float* susceptibility)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < popSize)
      susceptibility[tid] = phi[tickLevel[tid]];
  }

  template <class OP, bool zeroFirst>
  __global__ void
  _calcIntegral(const InfecIdx_t* infecIdx, const int infecSize,
		const CsrMatrix distance, float* eventTimes, 
		const int eventTimesPitch,const float* susceptibility,
		const float delta, const float omega, const float p, 
		const float nu, const float alpha1, const float alpha2, const float* integCache, float* output)
  {
    // Each warp calculates a row i of the sparse matrix

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int row = tid / 32; // Global Warp id
    int lane = tid & (32 - 1); // Id within a warp

    __shared__
      float buff[THREADSPERBLOCK];

    buff[threadIdx.x] = 0.0f;

    if (row < infecSize)
      {
        int i = infecIdx[row].ptr;

        int begin = distance.rowPtr[i];
        int end = distance.rowPtr[i + 1];
        float Ii = eventTimes[i];
        float Ri = eventTimes[i + eventTimesPitch * 2];

        float threadSum = 0.0f;
        for (int jj = begin + lane; jj < end; jj += 32)
          {
            // Integrated infection pressure
            float Ij = eventTimes[distance.colInd[jj]];
            float betaij = _H(fminf(Ri, Ij), fminf(Ii, Ij), nu, alpha1, alpha2, integCache);

            // Apply distance kernel and suscep
	    OP op;
            betaij *= p * op(distance.val[jj], delta, omega);
            betaij *= susceptibility[distance.colInd[jj]];
            threadSum += betaij;
          }
        buff[threadIdx.x] = threadSum;
      }

    // Reduce all warp sums and write to global memory.

    _shmemReduce(buff);

    if (threadIdx.x == 0)
      {
	if(zeroFirst)
	  output[blockIdx.x] = buff[0];
	else
	  output[blockIdx.x] += buff[0];
      }
  }



  __global__ void
  _bgIntegral(float* output, const float* eventTimes, const int popSize,
	      const float epsilon1,const float I1Time)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__
      float buff[THREADSPERBLOCK];
    buff[threadIdx.x] = 0.0f;

    if (tid < popSize)
      {
        float I = eventTimes[tid];
        buff[threadIdx.x] = epsilon1 * max(I - I1Time,0.0f);
      }

    _shmemReduce(buff);

    if(threadIdx.x == 0) output[blockIdx.x] = buff[0];
  }

  template <class OP, bool zeroFirst>
  __global__ void
  _calcProduct(const InfecIdx_t* infecIdx, const int infecSize,
	       const CsrMatrix distance, const float* eventTimes,
	       const int eventTimesPitch, const float* susceptibility,
	       const float epsilon1,
	       const float gamma1, const float delta, const float omega, const float p, const float nu,
	       const float alpha1, const float alpha2, float* prodCache)
  {
    // Each warp calculates a row of the sparse matrix

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int gwid = tid / 32; // Global Warp id
    int lane = tid & (32 - 1); // Id within a warp

    __shared__
      float threadProdCache[THREADSPERBLOCK];
    threadProdCache[threadIdx.x] = 0.0f;
    int row = gwid;

    if (row < infecSize)
      {
        int j = infecIdx[row].ptr;

        int begin = distance.rowPtr[j];
        int end = distance.rowPtr[j + 1];

        float Ij = eventTimes[j];

        for (int ii = begin + lane; ii < end; ii += 32)
          {
            int i = distance.colInd[ii];
            float Ii = eventTimes[i];
	    float Ri = eventTimes[eventTimesPitch * 2 + i];

            if (Ii < Ri)
              {
                float idxOnj = 0.0f;
                if (Ii < Ij and Ij <= Ri)
                  idxOnj += _h(Ij, Ii, nu, alpha1, alpha2);
		OP op;
                threadProdCache[threadIdx.x] += idxOnj * p * op(distance.valtr[ii],delta,omega);
              }
          }
        __syncthreads();

        // Reduce semi-products into productCache
        volatile float* vThreadProdCache = threadProdCache;
        if (lane < 16)
          {
            vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 16];
            vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 8];
            vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 4];
            vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 2];
            vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 1];
          }
        __syncthreads();

        // Write out to global memory
        if (lane == 0)
          {
	    float res = threadProdCache[threadIdx.x] * susceptibility[j]
	      * gamma1;
	    if(zeroFirst)
	      prodCache[j] = res + epsilon1;
	    else
	      prodCache[j] += res;
          }
      }
  }

  // __global__ void
  // _calcSpecPow(const unsigned int size, const int nSpecies, float* specpow,
  // 	       const int specpowPitch, const float* animals, const int animalsPitch,
  // 	       const float* powers)
  // {
  //   int row = blockIdx.x * blockDim.x + threadIdx.x;

  //   if (row < size)
  //     {
  //       for (unsigned int col = 0; col < nSpecies; ++col)
  //         {
  //           specpow[col * specpowPitch + row] = powf(
  // 						     animals[col * animalsPitch + row], powers[col]);
  //         }
  //     }
  // }

  template <class OP, bool zeroFirst>
  __global__ void
  _updateInfectionTimeIntegral(const unsigned int idx,
			       const InfecIdx_t* infecIdx, const float newTime, const CsrMatrix distance,
			       float* eventTimes, const int eventTimesPitch, const float* susceptibility,
			       const float delta, const float omega, const float p,
			       const float nu, const float alpha1, const float alpha2, const float* integCache, float* output)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__
      float buff[];
    buff[threadIdx.x] = 0.0f;

    int i = infecIdx[idx].ptr;
    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin)
      {
        int j = distance.colInd[begin + tid];

        float Ii = eventTimes[i];
        float Ri = eventTimes[i + eventTimesPitch * 2];

        float Ij = eventTimes[j];
        float Rj = eventTimes[j + eventTimesPitch * 2];
	
	OP op;

        float jOnIdx = 0.0f;
        if (Ij < Rj)
          {
            // Recalculate pressure from j on idx
            jOnIdx = _H(fminf(Rj, newTime), fminf(Ij, newTime), nu, alpha1, alpha2, integCache); // New pressure
            jOnIdx -= _H(fminf(Rj, Ii), fminf(Ii, Ij), nu, alpha1, alpha2, integCache); // Old pressure
	    jOnIdx *= op(distance.valtr[begin+tid],delta,omega);
	    // Apply infec and suscep
            jOnIdx *= susceptibility[i];
          }
	

        // Recalculate pressure from idx on j
        float IdxOnj = _H(fminf(Ri, Ij), fminf(newTime, Ij), nu, alpha1, alpha2, integCache);
        IdxOnj -= _H(fminf(Ri, Ij), fminf(Ii, Ij), nu, alpha1, alpha2, integCache);
        IdxOnj *= susceptibility[j];
	IdxOnj *= op(distance.val[begin+tid],delta,omega);
    
	buff[threadIdx.x] = (IdxOnj + jOnIdx) * p;

        // Reduce buffer into output
        _shmemReduce(buff);

      }

    if (threadIdx.x == 0)
      {
	if(zeroFirst) 
	  output[blockIdx.x] = buff[0];
	else
	  output[blockIdx.x] += buff[0];
      }
  }


  template <class OP, bool zeroFirst>
  __global__ void
  _updateInfectionTimeProduct(const unsigned int idx,
			      const InfecIdx_t* infecIdx, const float newTime, const CsrMatrix distance,
			      float* eventTimes, const int eventTimesPitch, const float* susceptibility,
			      const float epsilon1, const float gamma1,
			      const float delta, const float omega, const float p, 
			      const float nu, const float alpha1, const float alpha2, const int I1Idx, float* prodCache)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__
      float buff[];
    buff[threadIdx.x] = 0.0f;

    int i = infecIdx[idx].ptr;

    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin) // Massive amount of wasted time just here!
      {
        int j = distance.colInd[begin + tid];

        float Ij = eventTimes[j];
        float Nj = eventTimes[j + eventTimesPitch];

	OP op;

        if (Ij < Nj)
          {
            float Ii = eventTimes[i];
            float Ri = eventTimes[i + eventTimesPitch * 2];
            float Rj = eventTimes[j + eventTimesPitch * 2];

            // Adjust product cache from idx on others
            float idxOnj = 0.0f;
            if (Ii < Ij and Ij <= Ri)
              idxOnj -= _h(Ij, Ii, nu, alpha1, alpha2);

            if (newTime < Ij and Ij <= Ri)
              idxOnj += _h(Ij, newTime, nu, alpha1, alpha2);

            idxOnj *= gamma1 * susceptibility[j] * p * op(distance.val[begin+tid],delta, omega);
            prodCache[j] += idxOnj;

            // Recalculate instantaneous pressure on idx
            float jOnIdx = 0.0f;
            if (Ij < newTime and newTime <= Rj)
              jOnIdx = _h(newTime, Ij, nu, alpha1, alpha2);

            jOnIdx *= susceptibility[i] * p * op(distance.valtr[begin+tid],delta, omega);
            buff[threadIdx.x] = jOnIdx * gamma1;

          }

      }
        _shmemReduce(buff);

        if (threadIdx.x == 0)
          _atomicAdd(prodCache + i, buff[0]); // Maybe better to create an external reduction buffer here.
        if (tid == 0) {
	  if (zeroFirst)
	    _atomicAdd(prodCache + i, epsilon1);
        }
      
  }

  template <class OP,bool zeroFirst>
  __global__ void
  _addInfectionTimeIntegral(const unsigned int idx, const InfecIdx_t* infecIdx,
			    const float newTime, const CsrMatrix distance, const float* eventTimes,
			    const int eventTimesPitch, const float* susceptibility,
			    const float delta, const float omega, const float p,
			    const float nu, const float alpha1, const float alpha2, const float* integCache, float* output)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__
      float buff[];
    buff[threadIdx.x] = 0.0f;

    int i = infecIdx[idx].ptr;
    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin)
      {
        int j = distance.colInd[begin + tid];

        float Ii = eventTimes[i];
        float Ri = eventTimes[i + eventTimesPitch * 2];

        float Ij = eventTimes[j];
        float Rj = eventTimes[j + eventTimesPitch * 2];

	OP op;

        float jOnIdx = 0.0f;
        if (Ij < Rj)
          {
            // Calculate pressure from j on idx
            jOnIdx -= _H(fminf(Rj, Ii), fminf(Ij, Ii), nu, alpha1, alpha2, integCache);
            jOnIdx += _H(fminf(Rj, newTime), fminf(Ij, newTime), nu, alpha1, alpha2, integCache);
	    jOnIdx *= op(distance.valtr[begin+tid], delta, omega);
            // Apply infec and suscep
            jOnIdx *= susceptibility[i];
          }

        // Add pressure from idx on j
        float IdxOnj = _H(fminf(Ri, Ij), fminf(newTime, Ij), nu, alpha1, alpha2, integCache);
	IdxOnj *= op(distance.val[begin+tid], delta, omega);
        IdxOnj *= susceptibility[j];

        buff[threadIdx.x] = (IdxOnj + jOnIdx) * p;

        // Reduce buffer into output
        _shmemReduce(buff);
      }

    if (threadIdx.x == 0)
      {
	if (zeroFirst)
	  output[blockIdx.x] = buff[0];
	else
	  output[blockIdx.x] += buff[0];
      }
  }

  template <class OP, bool zeroFirst>
  __global__ void
  _delInfectionTimeIntegral(const unsigned int idx, const InfecIdx_t* infecIdx,
			    const float newTime, const CsrMatrix distance, float* eventTimes,
			    const int eventTimesPitch, const float* susceptibility,
			    const float delta, const float omega, const float p,
			    const float nu, const float alpha1, const float alpha2, const float* integCache, float* output)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__
      float buff[];
    buff[threadIdx.x] = 0.0f;

    int i = infecIdx[idx].ptr;
    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin)
      {
        int j = distance.colInd[begin + tid];

        float Ii = eventTimes[i];
        float Ri = eventTimes[i + eventTimesPitch * 2];

        float Ij = eventTimes[j];
        float Rj = eventTimes[j + eventTimesPitch * 2];

	OP op;

        float jOnIdx = 0.0f;
        if (Ij < Rj)
          {
            // Recalculate pressure from j on idx
            jOnIdx -= _H(fminf(Rj, Ii), fminf(Ii, Ij), nu, alpha1, alpha2, integCache); // Old pressure
            jOnIdx += _H(fminf(Rj, Ri), fminf(Ij, Ri), nu, alpha1, alpha2, integCache); // New pressure
	    jOnIdx *= op(distance.valtr[begin + tid], delta, omega);
	    // Apply infec and suscep
            jOnIdx *= susceptibility[i];
          }

        // Subtract pressure from idx on j
        float IdxOnj = 0.0f;
        IdxOnj -= _H(fminf(Ri, Ij), fminf(Ii, Ij), nu, alpha1, alpha2, integCache);
	IdxOnj *= op(distance.val[begin+tid], delta, omega);
        IdxOnj *= susceptibility[j];

        buff[threadIdx.x] = (IdxOnj + jOnIdx) * p;

        // Reduce buffer into output
        _shmemReduce(buff);

      }

    if (threadIdx.x == 0)
      {
	if(zeroFirst)
	  output[blockIdx.x] = buff[0];
	else
	  output[blockIdx.x] += buff[0];
      }
  }

  template <class OP, bool zeroFirst>
  __global__ void
  _addInfectionTimeProduct(const unsigned int idx, const InfecIdx_t* infecIdx,
			   const float newTime, const CsrMatrix distance, float* eventTimes,
			   const int eventTimesPitch, const float* susceptibility,
			   const float epsilon1, const float gamma1, const float delta, 
			   const float omega, const float p, const float nu, 
			   const float alpha1, const float alpha2, const int I1Idx, float* prodCache)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__
      float buff[];
    buff[threadIdx.x] = 0.0f;

    int i = infecIdx[idx].ptr;

    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin) // Massive amount of wasted time just here!
      {
        int j = distance.colInd[begin + tid];

        float Ij = eventTimes[j];
        float Nj = eventTimes[j + eventTimesPitch];

        if (Ij < Nj)
          { // Only look at infected individuals

            //float Ni = eventTimes[i + eventTimesPitch];
            float Ri = eventTimes[i + eventTimesPitch * 2];
            float Rj = eventTimes[j + eventTimesPitch * 2];

            // Adjust product cache from idx on others
            float idxOnj = 0.0f;
            if (newTime < Ij and Ij <= Ri)
              idxOnj += _h(Ij, newTime, nu, alpha1, alpha2);
	    
	    OP op;
            idxOnj *= gamma1 * susceptibility[j] * p * op(distance.val[begin+tid],delta, omega);
            prodCache[j] += idxOnj;

            // Calculate instantaneous pressure on idx
            float jOnIdx = 0.0f;
            if (Ij < newTime and newTime <= Rj)
              jOnIdx = _h(newTime, Ij, nu, alpha1, alpha2);

            jOnIdx *= gamma1 * susceptibility[i] * p * op(distance.valtr[begin+tid],delta,omega);

            buff[threadIdx.x] = jOnIdx;

          }
      }
        _shmemReduce(buff);

        if (threadIdx.x == 0)
          _atomicAdd(prodCache + i, buff[0]);
        if (tid == 0) { // Add background pressure, or turn to 1.0f for I1
	  if(zeroFirst)
	    _atomicAdd(prodCache + i, epsilon1);
        }
      
  }

  template <class OP, bool zeroFirst>
  __global__ void
  _delInfectionTimeProduct(const unsigned int idx, const InfecIdx_t* infecIdx,
			   const float newTime, const CsrMatrix distance, float* eventTimes,
			   const int eventTimesPitch, const float* susceptibility,
			   const float gamma1, const float delta, const float omega, 
			   const float p, const float nu, const float alpha1, const float alpha2,
			   float* prodCache)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    int i = infecIdx[idx].ptr;

    int begin = distance.rowPtr[i];
    int end = distance.rowPtr[i + 1];

    if (tid < end - begin) // Massive amount of wasted time just here!
      {
        int j = distance.colInd[begin + tid];

        float Ij = eventTimes[j];
	float Nj = eventTimes[j + eventTimesPitch];

        if (Ij < Nj)
          {
            float Ii = eventTimes[i];
	    float Ri = eventTimes[i + eventTimesPitch * 2];

            // Adjust product cache from idx on others
            float idxOnj = 0.0;
            if (Ii < Ij and Ij <= Ri)
              idxOnj -= _h(Ij, Ii, nu, alpha1, alpha2);

	    OP op;
            idxOnj *= gamma1 * susceptibility[j] * p * op(distance.val[begin+tid],delta,omega);
            prodCache[j] += idxOnj;
          }
      }
  }

  __global__
  void
  _knownInfectionsLikelihood(const InfecIdx_t* infecIdx,
			     const unsigned int knownInfecs, const float* eventTimes,
			     const int eventTimesPitch, const float a, const float b,
			     float* reductionBuff)
  {
    extern
      __shared__ float buff[];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    buff[threadIdx.x] = 0.0f;

    if (tid < knownInfecs)
      {
        int i = infecIdx[tid].ptr;
        float Ii = eventTimes[i];
        float Ni = eventTimes[eventTimesPitch + i];
        float d = Ni - Ii;
        buff[threadIdx.x] = logf(powf(b, a) * powf(d, a - 1) * expf(-d * b));
      }

    _shmemReduce(buff);

    if (threadIdx.x == 0)
      reductionBuff[blockIdx.x] = buff[0];
  }

  __global__
  void
  _knownInfectionsLikelihoodPNC(const InfecIdx_t* infecIdx,
				const unsigned int knownInfecs, const float* eventTimes,
				const int eventTimesPitch, const float a, const float oldGamma,
				const float newGamma, const float* rns, const float prob,
				float* reductionBuff)
  {
    extern
      __shared__ float buff[];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    buff[threadIdx.x] = 0.0f;

    if (tid < knownInfecs)
      {
	int i = infecIdx[tid].ptr;
	float Ii = eventTimes[i];
	float Ni = eventTimes[eventTimesPitch + i];
	float d = Ni - Ii;
        buff[threadIdx.x] = (powf(newGamma, prob) / powf(oldGamma, prob - 1.0f) - newGamma) * d
	  + a * (1 - prob) * log( newGamma / oldGamma );
      }

    _shmemReduce(buff);

    if (threadIdx.x == 0)
      reductionBuff[blockIdx.x] = buff[0];
  }

  __global__
  void
  _nonCentreInfecTimes(const InfecIdx_t* index, const int size,
		       float* eventTimes, int eventTimesPitch, const float factor,
		       const float* toCentre, const float prop)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
      {
	unsigned int i = index[tid].ptr;
	float notification = eventTimes[i + eventTimesPitch];
	float infection = eventTimes[i];
	eventTimes[i] = notification - (notification - infection) * powf(factor,prop);
      }
  }

  __global__
  void
  _collectInfectiousPeriods(const InfecIdx_t* index, const int size,
			    const float* eventTimes, const int eventTimesPitch, float* output)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
      {
        int i = index[tid].ptr;
        float infecPeriod = eventTimes[eventTimesPitch + i] - eventTimes[i];
        output[tid] = infecPeriod;
      }
  }

  __global__
  void
  _logTransform(const float* input, const int size, float* output)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
      output[tid] = logf(input[tid]);
  }

  __global__
  void
  _indirectedSum(const InfecIdx_t* index, const int size, const float* data,
		 float* output)
  {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    extern
      __shared__ float buff[];
    buff[threadIdx.x] = 0.0f;

    if (tid < size)
      {
        buff[threadIdx.x] = data[index[tid].ptr];
        _shmemReduce(buff);
      }
    if (threadIdx.x == 0)
      output[blockIdx.x] = buff[0];
  }

  float
  indirectedSum(const InfecIdx_t* index, const int size, const float* data)
  {
    int numBlocks = (size + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    thrust::device_vector<float> output(numBlocks);

    _indirectedSum<<<numBlocks, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(index, size, data, thrust::raw_pointer_cast(&output[0]));
    checkCudaError(cudaGetLastError());
    return thrust::reduce(output.begin(), output.end());

  }

  __global__
  void
  _reducePVectorStage1(float* input, const int size, const int I1Idx,
		       float* output)
  {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern
      __shared__ float buff[];
    buff[threadIdx.x] = 0.0f;

    if (tid < size)
      {
        if (tid == I1Idx)
          input[tid] = 1.0f; // Better put *after* our global memory fetch, I think!
        buff[threadIdx.x] = logf(input[tid]);

        //output[tid] = logf(input[tid]);
      }
    _shmemReduce(buff);
    if (threadIdx.x == 0)
      output[blockIdx.x] = buff[0];
  }

  void
  GpuLikelihood::ReduceProductVector()
  {
    // Reduces the device-side product vector into the device-side components struct

    cudaDeviceSynchronize();
    int blocksPerGrid = (devProduct_->size() + THREADSPERBLOCK - 1)
      / THREADSPERBLOCK;

    _reducePVectorStage1<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK * sizeof(float)>>>
      (thrust::raw_pointer_cast(&(*devProduct_)[0]),
       devProduct_->size(),
       I1Idx_,
       thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());

    cudaDeviceSynchronize();
    if(blocksPerGrid > 1) {
    CUDPPResult res = cudppReduce(addReduce_,
				  (float*) ((char*) devComponents_
					    + offsetof(LikelihoodComponents,logProduct)), thrust::raw_pointer_cast(&(*devWorkspace_)[0]), blocksPerGrid);
    if (res != CUDPP_SUCCESS)
      throw std::runtime_error(
			       "cudppReduce failed in GpuLikelihood::ReduceProductVector()");
    }
    else {
      checkCudaError(cudaMemcpy(&devComponents_->logProduct, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));
    }
    cudaDeviceSynchronize();

    //  float partial = thrust::reduce(devWorkspace_.begin(), devWorkspace_.begin() + devProduct_.size());
    //  checkCudaError(cudaMemcpy((float*)((char*)devComponents_ + offsetof(LikelihoodComponents,logProduct)), &partial, sizeof(float), cudaMemcpyHostToDevice));

  }

  GpuLikelihood::GpuLikelihood(PopDataImporter& population,
			       EpiDataImporter& epidemic, 
			       ContactDataImporter& contact, 
			       const size_t nSpecies, 
			       const float obsTime,
			       const float dLimit, 
			       const bool occultsOnlyDC, 
			       const int gpuId) : popSize_(0), 
						  numSpecies_(nSpecies), 
						  obsTime_(obsTime), 
						  I1Time_(0.0), 
						  I1Idx_(0), 
						  covariateCopies_(0), 
						  occultsOnlyDC_(occultsOnlyDC), 
						  movtBan_(obsTime)
  {

    // Get GPU details
    int deviceId;
    std::cout << "Trying GPGPU ID " << gpuId << std::endl;
    if(gpuId > -1) checkCudaError(cudaSetDevice(gpuId));
    checkCudaError(cudaGetDevice(&deviceId));
    cudaDeviceProp deviceProp;
    checkCudaError(cudaGetDeviceProperties(&deviceProp, deviceId));

#ifndef NDEBUG
    std::cout << "Using GPGPU: " << deviceProp.name << ", id " << deviceId
	      << ", located at PCI bus ID " << deviceProp.pciBusID << "\n";
#endif

    checkCudaError(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Allocate infec indicies
    hostInfecIdx_ = new thrust::host_vector<InfecIdx_t>;
    devInfecIdx_ = new thrust::device_vector<InfecIdx_t>;
    hostSuscOccults_ = new thrust::host_vector<InfecIdx_t>;

    // Load data into host memory
    LoadPopulation(population);
    LoadEpidemic(epidemic);
    SortPopulation();
    cerr << "Calculating distance matrix..." << flush;
    CalcDistanceMatrix(dLimit);
    cerr << "Done" << endl;
    cerr << "Loading contact matrix..." << flush;
    LoadContact(contact);
    cerr << "Done" << endl;

    // Set up on GPU
    SetSpecies();
    SetEvents();

    // Set up reference counter to covariate data
    covariateCopies_ = new size_t;
    *covariateCopies_ = 1;

    // Allocate product cache
    devProduct_ = new thrust::device_vector<float>;
    devProduct_->resize(maxInfecs_);
    thrust::fill(devProduct_->begin(), devProduct_->end(), 1.0f);

    // Allocate integral array
    int numRequiredThreads = maxInfecs_ * 32; // One warp per infection
    integralBuffSize_ = (numRequiredThreads + THREADSPERBLOCK - 1)
      / THREADSPERBLOCK;
    devWorkspace_ = new thrust::device_vector<float>;
    devWorkspace_->resize(maxInfecs_);

    // Components
    checkCudaError(
		   cudaHostAlloc((void**)&hostComponents_, sizeof(LikelihoodComponents), cudaHostAllocMapped));

    hostComponents_->bgIntegral = 0.0f;
    hostComponents_->integral = 0.0f;
    hostComponents_->sumI = 0.0f;
    hostComponents_->logProduct = 0.0f;

    checkCudaError(
		   cudaHostGetDevicePointer(&devComponents_, hostComponents_, 0));

    // Phi parameter array
    checkCudaError(cudaMalloc(&devPhi_, sizeof(float)*TICKLEVELS));
    
    // H function integral cache
    checkCudaError(cudaMalloc(&devHIntegCache_, sizeof(float)*5));

    // BLAS handles
    blasStat_ = cublasCreate(&cudaBLAS_);
    if (blasStat_ != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("CUBLAS init failed");

    sparseStat_ = cusparseCreate(&cudaSparse_);
    if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
      throw std::runtime_error("CUSPARSE init failed");

    sparseStat_ = cusparseCreateMatDescr(&crsDescr_);
    if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
      throw std::runtime_error("CUSPARSE matrix descriptor init failed");
    cusparseSetMatType(crsDescr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(crsDescr_, CUSPARSE_INDEX_BASE_ZERO);

    // RNG for non-centering
    curandStatus_t curandStatus = curandCreateGenerator(&cuRand_,
							CURAND_RNG_PSEUDO_DEFAULT);
    if (curandStatus != CURAND_STATUS_SUCCESS)
      {
        throw std::runtime_error("CURAND init failed");
      }
    curandStatus = curandSetPseudoRandomGeneratorSeed(cuRand_, 0ULL);
    if (curandStatus != CURAND_STATUS_SUCCESS)
      {
        throw std::runtime_error("Setting CURAND seed failed");
      }

    // CUDPP for faster reductions than Thrust (hopefully!)
    addReduceCfg_.op = CUDPP_ADD;
    addReduceCfg_.algorithm = CUDPP_REDUCE;
    addReduceCfg_.datatype = CUDPP_FLOAT;
    addReduceCfg_.options = CUDPP_OPTION_FORWARD;

    cudppCreate(&cudpp_);
    CUDPPResult res = cudppPlan(cudpp_, &addReduce_, addReduceCfg_, popSize_, 1,
				0);
    if (res != CUDPP_SUCCESS)
      {
        std::stringstream msg;
        msg << "CUDPP initialization failed with error " << res;
        throw std::runtime_error(msg.str().c_str());
      }

    // CUDPP for faster min reducitons
    minReduceCfg_.op = CUDPP_MIN;
    minReduceCfg_.algorithm = CUDPP_REDUCE;
    minReduceCfg_.datatype = CUDPP_FLOAT;
    minReduceCfg_.options = CUDPP_OPTION_FORWARD;

    //    res = cudppPlan(cudpp_, &minReduce_, minReduceCfg_, popSize_, 1, 0);
    //    if (res != CUDPP_SUCCESS)
    //      {
    //        std::stringstream msg;
    //        msg << "CUDPP initialization failed with error " << res;
    //        throw std::runtime_error(msg.str().c_str());
    //      }

#ifndef NDEBUG
    cerr << "ObsTime: " << obsTime_ << endl;
#endif

  }

  // Copy constructor
  GpuLikelihood::GpuLikelihood(const GpuLikelihood& other) :
    popSize_(other.popSize_), 
    numKnownInfecs_(other.numKnownInfecs_), 
    maxInfecs_(other.maxInfecs_), 
    numSpecies_(other.numSpecies_), 
    hostPopulation_(other.hostPopulation_), 
    obsTime_(other.obsTime_),
    I1Time_(other.I1Time_), 
    I1Idx_(other.I1Idx_), 
    covariateCopies_(other.covariateCopies_), 
    devAnimals_(other.devAnimals_), 
    animalsPitch_(other.animalsPitch_), 
    devD_(other.devD_), 
    devC_(other.devC_),
    hostDRowPtr_(other.hostDRowPtr_), 
    hostCRowPtr_(other.hostCRowPtr_),
    dnnz_(other.dnnz_),
    integralBuffSize_(other.integralBuffSize_), 
    epsilon1_(other.epsilon1_),
    gamma1_(other.gamma1_), 
    delta_(other.delta_), 
    omega_(other.omega_), 
    beta1_(other.beta1_), 
    beta2_(other.beta2_), 
    nu_(other.nu_), 
    alpha1_(other.alpha1_),
    alpha2_(other.alpha2_), 
    a_(other.a_), 
    b_(other.b_), 
    cuRand_(other.cuRand_)
  {
    timeval start, end;
    gettimeofday(&start, NULL);
    // Allocate Animals_
    checkCudaError(
		   cudaMallocPitch(&devAnimalsInfPow_, &animalsInfPowPitch_, maxInfecs_ * sizeof(float), numSpecies_));
    animalsInfPowPitch_ /= sizeof(float);
    checkCudaError(
		   cudaMemcpy2D(devAnimalsInfPow_,animalsInfPowPitch_*sizeof(float),other.devAnimalsInfPow_,other.animalsInfPowPitch_*sizeof(float),maxInfecs_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));

    checkCudaError(
		   cudaMallocPitch(&devAnimalsSuscPow_, &animalsSuscPowPitch_, popSize_ * sizeof(float), numSpecies_));
    animalsSuscPowPitch_ /= sizeof(float);
    checkCudaError(
		   cudaMemcpy2D(devAnimalsSuscPow_,animalsSuscPowPitch_*sizeof(float),other.devAnimalsSuscPow_,other.animalsSuscPowPitch_*sizeof(float),popSize_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));

    // Allocate and copy event times - popSize_ * NUMEVENTS matrix
    checkCudaError(
		   cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS));
    eventTimesPitch_ /= sizeof(float);
    checkCudaError(
		   cudaMemcpy2D(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),popSize_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

    // Allocate and copy intermediate infectivity and susceptibility
    checkCudaError(cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float)));
    checkCudaError(
		   cudaMemcpy(devSusceptibility_, other.devSusceptibility_, popSize_ * sizeof(float),cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMalloc(&devInfectivity_, maxInfecs_ * sizeof(float)));
    checkCudaError(
		   cudaMemcpy(devInfectivity_, other.devInfectivity_, maxInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

    checkCudaError(cudaMalloc(&devPhi_, sizeof(float)*TICKLEVELS));

    // Infection index
    devInfecIdx_ = new thrust::device_vector<InfecIdx_t>(*other.devInfecIdx_);
    hostInfecIdx_ = new thrust::host_vector<InfecIdx_t>(*other.hostInfecIdx_);

    hostSuscOccults_ = new thrust::host_vector<InfecIdx_t>(*other.hostSuscOccults_);

    // Allocate and copy likelihood components;
    devProduct_ = new thrust::device_vector<float>(*(other.devProduct_));
    devWorkspace_ = new thrust::device_vector<float>;
    devWorkspace_->resize(other.devWorkspace_->size());
    logLikelihood_ = other.logLikelihood_;

    checkCudaError(
		   cudaHostAlloc(&hostComponents_, sizeof(LikelihoodComponents), cudaHostAllocMapped));

    *hostComponents_ = *(other.hostComponents_);

    checkCudaError(
		   cudaHostGetDevicePointer(&devComponents_, hostComponents_, 0));

    // Phi parameters
    phi_ = other.phi_;
    
    // H function integral cache
    checkCudaError(cudaMalloc(&devHIntegCache_, sizeof(float)*5));

    // Refresh parameters
    RefreshParameters();
    
    // BLAS handles
    blasStat_ = other.blasStat_;
    cudaBLAS_ = other.cudaBLAS_;
    sparseStat_ = other.sparseStat_;
    cudaSparse_ = other.cudaSparse_;
    crsDescr_ = other.crsDescr_;

    ++*covariateCopies_; // Increment copies of covariate data

    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;

    // CUDAPP bits
    // CUDPP for faster reductions than Thrust (hopefully!)
    addReduceCfg_ = other.addReduceCfg_;
    cudpp_ = other.cudpp_;
    addReduce_ = other.addReduce_;

    cudaDeviceSynchronize();

  }

  // Assignment constructor
  const GpuLikelihood&
  GpuLikelihood::operator=(const GpuLikelihood& other)
  {
    //  timeval start, end;
    //  gettimeofday(&start, NULL);
    // Copy animal powers
    checkCudaError(
		   cudaMemcpy2DAsync(devAnimalsInfPow_,animalsInfPowPitch_*sizeof(float),other.devAnimalsInfPow_,other.animalsInfPowPitch_*sizeof(float),maxInfecs_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));
    checkCudaError(
		   cudaMemcpy2DAsync(devAnimalsSuscPow_,animalsSuscPowPitch_*sizeof(float),other.devAnimalsSuscPow_,other.animalsSuscPowPitch_*sizeof(float),popSize_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));

    // copy event times
    checkCudaError(
		   cudaMemcpy2DAsync(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),popSize_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

    // copy intermediate infectivity and susceptibility
    checkCudaError(
		   cudaMemcpyAsync(devSusceptibility_, other.devSusceptibility_, popSize_ * sizeof(float),cudaMemcpyDeviceToDevice));
    checkCudaError(
		   cudaMemcpyAsync(devInfectivity_, other.devInfectivity_, maxInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

    // Infection index
    *devInfecIdx_ = *(other.devInfecIdx_);
    *hostInfecIdx_ = *(other.hostInfecIdx_);

    // Internals
    I1Idx_ = other.I1Idx_;
    I1Time_ = other.I1Time_;
    *hostSuscOccults_ = *other.hostSuscOccults_;

    // Host Parameters Copy
    epsilon1_ = other.epsilon1_;
    gamma1_ = other.gamma1_;
    phi_ = other.phi_;
    delta_ = other.delta_;
    omega_ = other.omega_;
    beta1_ = other.beta1_;
    beta2_ = other.beta2_;
    nu_ = other.nu_;
    alpha1_ = other.alpha1_;
    alpha2_ = other.alpha2_;
    a_ = other.a_;
    b_ = other.b_;

    RefreshParameters();

    // Likelihood components
    // copy product vector
    *devProduct_ = *other.devProduct_;

    *hostComponents_ = *other.hostComponents_;
    logLikelihood_ = other.logLikelihood_;

    //  gettimeofday(&end, NULL);
    //  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
    //      << timeinseconds(start, end) << std::endl;

    cudaDeviceSynchronize();
    return *this;
  }

  void
  GpuLikelihood::InfecCopy(const GpuLikelihood& other)
  {

    // copy event times
    checkCudaError(
		   cudaMemcpy2DAsync(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),maxInfecs_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

    // Infection index
    *devInfecIdx_ = *other.devInfecIdx_;
    *hostInfecIdx_ = *other.hostInfecIdx_;

    // Internals
    I1Idx_ = other.I1Idx_;
    I1Time_ = other.I1Time_;
    *hostSuscOccults_ = *other.hostSuscOccults_;

    // copy product vector
    *devProduct_ = *other.devProduct_;

    // Likelihood components
    *hostComponents_ = *other.hostComponents_;
    logLikelihood_ = other.logLikelihood_;

    cudaDeviceSynchronize();

  }

  GpuLikelihood::~GpuLikelihood()
  {

    // Destroy non-shared members first
    if (devEventTimes_)
      cudaFree(devEventTimes_);
    if (devAnimalsSuscPow_)
      cudaFree(devAnimalsSuscPow_);
    if (devAnimalsInfPow_)
      cudaFree(devAnimalsInfPow_);
    if (devSusceptibility_)
      cudaFree(devSusceptibility_);
    if (devInfectivity_)
      cudaFree(devInfectivity_);
    if(devPhi_)
      cudaFree(devPhi_);
    if(devHIntegCache_)
      cudaFree(devHIntegCache_);

    if (hostComponents_)
      cudaFreeHost(hostComponents_);

    // Destroy Thrust bits
    if(hostInfecIdx_) delete hostInfecIdx_;
    if(devInfecIdx_) delete devInfecIdx_;
    if(hostSuscOccults_) delete hostSuscOccults_;
    if(devProduct_) delete devProduct_;
    if(devWorkspace_) delete devWorkspace_;

    // Choose whether to destroy shared members
    if (*covariateCopies_ == 1) // We're the last copy to be destroyed
      {
        cudaFree(devAnimals_);
        destroyCsrMatrix(devD_);
	destroyCsrMatrix(devC_);
        delete[] hostDRowPtr_;
	delete[] hostCRowPtr_;
        cublasDestroy(cudaBLAS_);
        cusparseDestroy(cudaSparse_);
        curandDestroyGenerator(cuRand_);
        delete covariateCopies_;

        cudppDestroyPlan(addReduce_);
        cudppDestroy(cudpp_);

	if(crsDescr_)
	  cusparseDestroyMatDescr(crsDescr_);
	
	cudaDeviceSynchronize();
	checkCudaError(cudaDeviceReset());
      }
    else
      {
        --(*covariateCopies_);
      }


  }

  void
  GpuLikelihood::CalcDistanceMatrix(const float dLimit)
  {
    float2* coords = new float2[popSize_];
    Population::iterator it = hostPopulation_.begin();
    for (size_t i = 0; i < popSize_; ++i)
      {
        coords[i] = make_float2((float) it->x, (float) it->y);
        it++;
      }

    devD_ = makeSparseDistance(coords, popSize_, dLimit);
    dnnz_ = devD_->nnz;

    cerr << "About to allocate hostDRowPtr" << endl;
    hostDRowPtr_ = new int[popSize_ + 1];
    cerr << "Allocated hostDRowPtr_ " << endl;
    checkCudaError(
		   cudaMemcpy(hostDRowPtr_, devD_->rowPtr, (popSize_+1)*sizeof(int), cudaMemcpyDeviceToHost));

    delete[] coords;
  }

  void
  GpuLikelihood::SetEvents()
  {

    // Set up Species and events
    float* eventsMatrix = new float[popSize_ * NUMEVENTS];
    Population::iterator it = hostPopulation_.begin();
    for (size_t i = 0; i < popSize_; ++i)
      {
        eventsMatrix[i] = it->I;
        eventsMatrix[i + popSize_] = it->N;
        eventsMatrix[i + popSize_ * 2] = it->R;
        ++it;
      }

    // Allocate Event times - popSize_ * NUMEVENTS matrix
    checkCudaError(
		   cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS));
    eventTimesPitch_ /= sizeof(float);

    // Get event times into GPU memory
    cudaError_t rv = cudaMemcpy2D(devEventTimes_,
				  eventTimesPitch_ * sizeof(float), eventsMatrix,
				  popSize_ * sizeof(float), popSize_ * sizeof(float), NUMEVENTS,
				  cudaMemcpyHostToDevice);
    if (rv != cudaSuccess)

      throw GpuRuntimeError("Copying event times to device failed", rv);

    // Set any event times greater than obsTime to obsTime
    int blocksPerGrid = (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    _sanitizeEventTimes<<<blocksPerGrid, THREADSPERBLOCK>>>(devEventTimes_, eventTimesPitch_, obsTime_, popSize_);
    checkCudaError(cudaGetLastError());

    thrust::device_ptr<float> p(devEventTimes_);
    hostInfecIdx_->clear();
    for (size_t i = 0; i < numKnownInfecs_; ++i)
      {
        hostInfecIdx_->push_back(i);
      }
    *devInfecIdx_ = *hostInfecIdx_;

    delete[] eventsMatrix;
  }

  void
  GpuLikelihood::SetSpecies()
  {

    // Set up Species and events
    int* speciesMatrix = new int[popSize_ * numSpecies_];
    Population::const_iterator it = hostPopulation_.begin();
    for (size_t i = 0; i < hostPopulation_.size(); ++i)
      {
	speciesMatrix[i] = (int)it->ticks;
        ++it;
      }

    // Allocate Animals_
    checkCudaError(
		   cudaMallocPitch(&devAnimals_, &animalsPitch_, popSize_ * sizeof(int), numSpecies_));
    animalsPitch_ /= sizeof(float);
    checkCudaError(
		   cudaMallocPitch(&devAnimalsSuscPow_, &animalsSuscPowPitch_, popSize_ * sizeof(float), numSpecies_));
    animalsSuscPowPitch_ /= sizeof(float);
    checkCudaError(
		   cudaMallocPitch(&devAnimalsInfPow_, &animalsInfPowPitch_, maxInfecs_ * sizeof(float), numSpecies_));
    animalsInfPowPitch_ /= sizeof(float);

    // Allocate intermediate infectivity and susceptibility
    checkCudaError(cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float)));
    checkCudaError(cudaMalloc(&devInfectivity_, maxInfecs_ * sizeof(float)));

    cudaError_t rv = cudaMemcpy2D(devAnimals_, animalsPitch_ * sizeof(int),
				  speciesMatrix, popSize_ * sizeof(int), popSize_ * sizeof(int),
				  numSpecies_, cudaMemcpyHostToDevice);
    if (rv != cudaSuccess)
      throw GpuRuntimeError("Failed copying species data to device", rv);

    delete[] speciesMatrix;

  }

  void
  GpuLikelihood::SetDistance(const float* data, const int* rowptr,
			     const int* colind)
  {

    checkCudaError(cudaMalloc(&devD_->val, dnnz_ * sizeof(float)));
    checkCudaError(cudaMalloc(&devD_->rowPtr, (maxInfecs_ + 1) * sizeof(int)));
    checkCudaError(cudaMalloc(&devD_->colInd, dnnz_ * sizeof(float)));

    hostDRowPtr_ = new int[maxInfecs_ + 1];

    checkCudaError(
		   cudaMemcpy(devD_->val, data, dnnz_ * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(
		   cudaMemcpy(devD_->rowPtr, rowptr, (maxInfecs_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(
		   cudaMemcpy(devD_->colInd, colind, dnnz_ * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(
		   cudaMemcpy(hostDRowPtr_, rowptr, (maxInfecs_ + 1)*sizeof(int), cudaMemcpyHostToHost));
  }


  inline
  void
  GpuLikelihood::UpdateI1()
  {
    thrust::device_vector<InfecIdx_t>::iterator myMin;
    myMin = thrust::min_element(devInfecIdx_->begin(), devInfecIdx_->end(),
				IndirectMin<float>(devEventTimes_));
    InfecIdx_t tmp = *myMin;
    I1Idx_ = tmp.ptr;

    thrust::device_ptr<float> v(devEventTimes_);
    I1Time_ = v[I1Idx_];
    cudaDeviceSynchronize();
  }

  inline
  void
  GpuLikelihood::CalcSusceptibility()
  {
    int numBlocks = (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    _calcSusceptibility<<<numBlocks, THREADSPERBLOCK>>>(devAnimals_,popSize_,devPhi_,devSusceptibility_);
  }

  inline
  void
  GpuLikelihood::CalcBgIntegral()
  {
    int numBlocks = (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    _bgIntegral<<<numBlocks, THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devWorkspace_)[0]), devEventTimes_, popSize_, *epsilon1_, I1Time_);

    cudaDeviceSynchronize();
    if(numBlocks > 1) {
    CUDPPResult res = cudppReduce(addReduce_, &devComponents_->bgIntegral,
				  thrust::raw_pointer_cast(&(*devWorkspace_)[0]), numBlocks);
    if(res != CUDPP_SUCCESS) throw logic_error("CUDPP failed");
    }
    else {
      checkCudaError(cudaMemcpy(&devComponents_->bgIntegral, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    if(hostComponents_->bgIntegral <= 0.0f) cerr << "bgIntegral = " << hostComponents_->bgIntegral << endl;
    assert(hostComponents_->bgIntegral >= 0.0f);
#endif
  }

  inline
  void
  GpuLikelihood::CalcProduct()
  {

    _calcProduct<DistanceKernel,true><<<integralBuffSize_,THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
									     devInfecIdx_->size(),*devD_,
									     devEventTimes_,eventTimesPitch_,
									     devSusceptibility_,
									     *epsilon1_, *gamma1_,*delta_,*omega_,
									     *beta1_,*nu_, *alpha1_, *alpha2_, 
									     thrust::raw_pointer_cast(&(*devProduct_)[0]));
    _calcProduct<Identity,false><<<integralBuffSize_,THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
    									devInfecIdx_->size(),*devC_,
    									devEventTimes_,eventTimesPitch_,
    									devSusceptibility_,
    									*epsilon1_,*gamma1_,*delta_,*omega_,
    									*beta2_,*nu_,*alpha1_, *alpha2_, 
    									thrust::raw_pointer_cast(&(*devProduct_)[0]));
    checkCudaError(cudaGetLastError());

    ReduceProductVector();
  }

  inline
  void
  GpuLikelihood::CalcIntegral()
  {
    int numRequiredThreads = devInfecIdx_->size() * 32; // One warp per infection
    int integralBuffSize = (numRequiredThreads + THREADSPERBLOCK - 1)
      / THREADSPERBLOCK;

    _calcIntegral<DistanceKernel,true><<<integralBuffSize,THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
									     devInfecIdx_->size(),*devD_,
									     devEventTimes_,eventTimesPitch_,
									     devSusceptibility_,
									     *delta_,*omega_,*beta1_,*nu_,*alpha1_, *alpha2_, devHIntegCache_,
									     thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());

    _calcIntegral<Identity,false><<<integralBuffSize,THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
    									devInfecIdx_->size(),*devC_,
    									devEventTimes_,eventTimesPitch_,
    									devSusceptibility_,
    									*delta_,*omega_,*beta2_,*nu_, *alpha1_, *alpha2_, devHIntegCache_,
    									thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());
    
    
    if(integralBuffSize > 1) {
      CUDPPResult res = cudppReduce(addReduce_, &devComponents_->integral,
				    thrust::raw_pointer_cast(&(*devWorkspace_)[0]), integralBuffSize);
      if (res != CUDPP_SUCCESS)
        throw std::runtime_error(
				 "cudppReduce failed in GpuLikelihood::CalcIntegral()");
    }
    else checkCudaError(cudaMemcpy(&devComponents_->integral, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));

  }

  void
  GpuLikelihood::FullCalculate()
  {

#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    RefreshParameters();
    UpdateI1();
    CalcSusceptibility();
    //CheckSuscep();
    CalcIntegral();
    CalcProduct();
    CalcBgIntegral();

    cudaDeviceSynchronize();
    hostComponents_->integral *= *gamma1_;
    logLikelihood_ = hostComponents_->logProduct
      - (hostComponents_->integral + hostComponents_->bgIntegral);


#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;
    std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
	      << std::endl;
#endif

#ifndef NDEBUG
    cerr << __FUNCTION__ << " (likelihood)\n";
    PrintLikelihoodComponents();
    PrintParameters();
    PrintEventTimes();
    cerr << endl;
#endif
  }

  void
  GpuLikelihood::Calculate()
  {
#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    RefreshParameters();
    UpdateI1();
    CalcSusceptibility();
    CalcIntegral();
    CalcProduct();
    CalcBgIntegral();

    cudaDeviceSynchronize();
    hostComponents_->integral *= *gamma1_;
    logLikelihood_ = hostComponents_->logProduct
      - (hostComponents_->integral + hostComponents_->bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;
#endif

#ifndef NDEBUG
    cerr << __FUNCTION__ << " (likelihood)\n";
    PrintLikelihoodComponents();
    PrintParameters();
    PrintEventTimes();
    cerr << endl;
#endif
  }

  float
  GpuLikelihood::InfectionPart()
  {
    int blocksPerGrid = (GetNumKnownInfecs() + THREADSPERBLOCK - 1)
      / THREADSPERBLOCK;

    _knownInfectionsLikelihood<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
												  GetNumKnownInfecs(), devEventTimes_, eventTimesPitch_, *a_, *b_, thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());

    float loglikelihood = 0.0f;

    for (size_t i = GetNumKnownInfecs(); i < this->GetNumInfecs(); ++i)
      {
        float Ii, Ni;
        checkCudaError(
		       cudaMemcpy(&Ii, devEventTimes_+(*hostInfecIdx_)[i].ptr, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaError(
		       cudaMemcpy(&Ni, devEventTimes_+eventTimesPitch_+(*hostInfecIdx_)[i].ptr, sizeof(float), cudaMemcpyDeviceToHost));
        loglikelihood += log(
			     gsl_cdf_gamma_Q(Ni - Ii, (float) *a_, 1.0 / (float) *b_));
      }

    loglikelihood += thrust::reduce(devWorkspace_->begin(),
				    devWorkspace_->begin() + blocksPerGrid);

    return loglikelihood;
  }

  void
  GpuLikelihood::UpdateInfectionTime(const unsigned int idx, const float inTime)
  {
    // Require to know number of cols per row -- probably store in host mem.
    // Also, may be optimal to use a much lower THREADSPERBLOCK than the app-wide setting.

#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    if (idx >= hostInfecIdx_->size())
      throw std::range_error(
			     "Invalid idx in GpuLikelihood::UpdateInfectionTime");

    // Save likelihood components
    float savedIntegral = hostComponents_->integral;

    int i = (*hostInfecIdx_)[idx].ptr;

    thrust::device_ptr<float> eventTimesPtr(devEventTimes_);
    float newTime = hostPopulation_[i].N - inTime; // Relies on hostPopulation.N *NOT* being changed!
    float oldTime = eventTimesPtr[i];

    bool haveNewI1 = false;
    if (newTime < I1Time_ or i == I1Idx_)
      {
        haveNewI1 = true;
        (*devProduct_)[I1Idx_] = *epsilon1_;
      }

    (*devProduct_)[i] = 0.0f;

    int blocksPerGridD = (hostDRowPtr_[i + 1] - hostDRowPtr_[i] + THREADSPERBLOCK
			 - 1) / THREADSPERBLOCK + 1;

    int blocksPerGridC = (hostCRowPtr_[i+1] - hostCRowPtr_[i] + THREADSPERBLOCK - 1) / THREADSPERBLOCK + 1;

    int maxBpG = max(blocksPerGridD, blocksPerGridC);

    thrust::fill(devWorkspace_->begin(),devWorkspace_->end(),0.0f);

#ifndef NDEBUG
    cerr << "Moving idx " << idx << " from " <<  oldTime << " to " << newTime << endl;
#endif

    // Integrated infection pressure
    _updateInfectionTimeIntegral<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, 
															  thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
															  newTime,
															  *devD_,
															  devEventTimes_, 
															  eventTimesPitch_, 
															  devSusceptibility_,
															  *delta_, 
															  *omega_, 
															  *beta1_, 
															  *nu_, 
															  *alpha1_, *alpha2_,
															  devHIntegCache_,
															  thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    _updateInfectionTimeIntegral<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, 
														     thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
														     newTime,
														     *devC_,
														     devEventTimes_, 
														     eventTimesPitch_, 
														     devSusceptibility_,
														     *delta_, 
														     *omega_, 
														     *beta2_, 
														     *nu_, 
														     *alpha1_, *alpha2_, devHIntegCache_, 
														     thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();
    if(maxBpG > 1) {
      CUDPPResult res = cudppReduce(addReduce_, &devComponents_->integral,
				    thrust::raw_pointer_cast(&(*devWorkspace_)[0]), maxBpG);
      if (res != CUDPP_SUCCESS)
	throw std::runtime_error(
				 "cudppReduce failed in GpuLikelihood::UpdateInfectionTime()");
    }
    else {
      checkCudaError(cudaMemcpy(&devComponents_->integral, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));
#ifndef NDEBUG
      cerr << __FUNCTION__ << ": blocksPerGrid = " << blocksPerGridD << endl;
#endif
    }

    _updateInfectionTimeProduct<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, 
												   thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
												   newTime, 
												   *devD_,
												   devEventTimes_, 
												   eventTimesPitch_,
												   devSusceptibility_, 
												   *epsilon1_, 
												   *gamma1_, 
												   *delta_, 
												   *omega_, 
												   *beta1_,
												   *nu_, 
															 *alpha1_, *alpha2_, 
												   I1Idx_, 
												   thrust::raw_pointer_cast(&(*devProduct_)[0]));

     _updateInfectionTimeProduct<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, 
     												   thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
     												   newTime, 
     												   *devC_,
     												   devEventTimes_, 
     												   eventTimesPitch_,
     												   devSusceptibility_, 
     												   *epsilon1_, 
     												   *gamma1_, 
     												   *delta_, 
     												   *omega_, 
     												   *beta2_,
     												   *nu_, 
														     *alpha1_,*alpha2_, 
     												   I1Idx_, 
     												   thrust::raw_pointer_cast(&(*devProduct_)[0]));
    checkCudaError(cudaGetLastError());

    // Make the change to the population
    eventTimesPtr[i] = newTime;

    if (haveNewI1)
      {
        UpdateI1();
        CalcBgIntegral();
        haveNewI1 = true;
#ifndef NDEBUG
        std::cerr << "New I1" << std::endl;
#endif
      }

    ReduceProductVector();

    // Collect results and update likelihood

    //checkCudaError(cudaMemcpy(&localUpdate, devComponents_, sizeof(LikelihoodComponents), cudaMemcpyDeviceToHost)); // CUDA_MEMCPY
    cudaDeviceSynchronize();
    hostComponents_->integral = savedIntegral
      + hostComponents_->integral * *gamma1_;
    if (!haveNewI1) {
      hostComponents_->bgIntegral += *epsilon1_ * (newTime - oldTime);
    }

    logLikelihood_ = hostComponents_->logProduct
      - (hostComponents_->integral + hostComponents_->bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;
    std::cerr.precision(20);
    std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
	      << std::endl;
    std::cerr << "I1: " << I1Idx_ << " at " << I1Time_ << std::endl;
#endif

#ifndef NDEBUG
    cerr << __FUNCTION__ << " (likelihood)\n";
    PrintLikelihoodComponents();
    PrintParameters();
    PrintEventTimes();
    cerr << endl;
#endif
  }

  void
  GpuLikelihood::AddInfectionTime(const unsigned int idx, const float inTime)
  {
    // idx is the position in the hostSuscOccults vector (ie the idx'th occult)
    // inTime is the proposed Ni - Ii time

#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    if (idx >= hostSuscOccults_->size())
      throw std::range_error("Invalid idx in GpuLikelihood::AddInfectionTime");

    // Save likelihood components
    float savedIntegral = hostComponents_->integral;

    unsigned int i = (*hostSuscOccults_)[idx].ptr;

    thrust::device_ptr<float> eventTimesPtr(devEventTimes_);
    float Ni = hostPopulation_[i].N;
    float newTime = Ni - inTime;

    // Update the indices
    devInfecIdx_->push_back(i);
    hostInfecIdx_->push_back(i);
    hostSuscOccults_->erase(hostSuscOccults_->begin() + idx);

    (*devProduct_)[i] = 0.0f; // Zero out product cache
    bool haveNewI1 = false;
    if(newTime < I1Time_) {
      (*devProduct_)[I1Idx_] = *epsilon1_;
      haveNewI1 = true;
    }

    unsigned int addIdx = devInfecIdx_->size() - 1;

    int blocksPerGridD = (hostDRowPtr_[i + 1] - hostDRowPtr_[i] + THREADSPERBLOCK
			 - 1) / THREADSPERBLOCK + 1;
    int blocksPerGridC = (hostCRowPtr_[i + 1] - hostCRowPtr_[i] + THREADSPERBLOCK - 1) / THREADSPERBLOCK + 1;
    int maxBpG = max(blocksPerGridD,blocksPerGridC);

    thrust::fill(devWorkspace_->begin(), devWorkspace_->end(), 0.0f);

    _addInfectionTimeIntegral<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, 
														 thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														 newTime,
														 *devD_, 
														 devEventTimes_, 
														 eventTimesPitch_, 
														 devSusceptibility_,
														 *delta_, 
														 *omega_, 
														 *beta1_,
														 *nu_, 
														       *alpha1_, *alpha2_,
														       devHIntegCache_, 
														 thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    _addInfectionTimeIntegral<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, 
														 thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														 newTime,
														 *devC_, 
														 devEventTimes_, 
														 eventTimesPitch_, 
														 devSusceptibility_,
														 *delta_, 
														 *omega_, 
														 *beta2_,
														 *nu_, 
														  *alpha1_,*alpha2_, 
														  devHIntegCache_,
														 thrust::raw_pointer_cast(&(*devWorkspace_)[0]));

    checkCudaError(cudaGetLastError());
    if(maxBpG > 1) {
      CUDPPResult res = cudppReduce(addReduce_, &devComponents_->integral,
				    thrust::raw_pointer_cast(&(*devWorkspace_)[0]), maxBpG);
      if (res != CUDPP_SUCCESS)
	throw std::runtime_error(
				 "cudppReduce failed in GpuLikelihood::UpdateInfectionTime()");
    }
    else {
      checkCudaError(cudaMemcpy(&devComponents_->integral, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));
#ifndef NDEBUG
      cerr << __FUNCTION__ << ": blocksPerGrid = " << blocksPerGridD << endl;
#endif
    }

    _addInfectionTimeProduct<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, 
														thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
														newTime,
														*devD_, 
														devEventTimes_,
														eventTimesPitch_,
														devSusceptibility_,
														*epsilon1_,
														*gamma1_,
														*delta_, 
														*omega_, 
														*beta1_,
														*nu_, 
														      *alpha1_,*alpha2_, 
														I1Idx_, 
														thrust::raw_pointer_cast(&(*devProduct_)[0]));
    _addInfectionTimeProduct<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, 
    														thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
    														newTime,
    														*devC_, 
    														devEventTimes_,
    														eventTimesPitch_,
    														devSusceptibility_,
    														*epsilon1_,
    														*gamma1_,
    														*delta_, 
    														*omega_, 
    														*beta2_,
    														*nu_, 
														 *alpha1_,*alpha2_, 
    														I1Idx_, 
    														thrust::raw_pointer_cast(&(*devProduct_)[0]));
    checkCudaError(cudaGetLastError());

    // Update the population
    eventTimesPtr[i] = newTime;
    if (haveNewI1)
      {
        UpdateI1();
        CalcBgIntegral();
#ifndef NDEBUG
        std::cerr << "New I1" << std::endl;
#endif
      }

    ReduceProductVector();

    // Collect results and update likelihood
    cudaDeviceSynchronize();
    hostComponents_->integral = savedIntegral
      + hostComponents_->integral * *gamma1_;
    if (!haveNewI1) {
      hostComponents_->bgIntegral += *epsilon1_ * (newTime - Ni);
    }

    logLikelihood_ = hostComponents_->logProduct
      - (hostComponents_->integral + hostComponents_->bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;
#endif

#ifndef NDEBUG
    cerr << __FUNCTION__ << " (likelihood)\n";
    PrintLikelihoodComponents();
    PrintParameters();
    PrintEventTimes();
    cerr << endl;
#endif
  }

  void
  GpuLikelihood::DeleteInfectionTime(const unsigned int idx)
  {
    // Delete the idx'th occult ( = idx+numKnownInfecs_ infective)

#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    // Range check
    if (idx >= devInfecIdx_->size() - numKnownInfecs_)
      throw std::range_error(
			     "Invalid idx in GpuLikelihood::DeleteInfectionTime");

    // Save likelihood components
    float savedIntegral = hostComponents_->integral;

    // Identify occult to delete
    unsigned int ii = idx + numKnownInfecs_;
    unsigned int i = (*hostInfecIdx_)[ii].ptr;

    thrust::device_ptr<float> eventTimesPtr(devEventTimes_);

    float notification = hostPopulation_[i].N;
    float oldI = eventTimesPtr[i];

    int blocksPerGridD = (hostDRowPtr_[i + 1] - hostDRowPtr_[i] + THREADSPERBLOCK
			 - 1) / THREADSPERBLOCK + 1;
    int blocksPerGridC = (hostCRowPtr_[i + 1] - hostCRowPtr_[i] + THREADSPERBLOCK
			 - 1) / THREADSPERBLOCK + 1;
    int maxBpG = max(blocksPerGridD, blocksPerGridC);

    thrust::fill(devWorkspace_->begin(), devWorkspace_->end(), 0.0f);
    _delInfectionTimeIntegral<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(ii, 
														 thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														 notification,
														 *devD_,
														 devEventTimes_, 
														 eventTimesPitch_, 
														 devSusceptibility_,
														 *delta_, 
														 *omega_, 
														 *beta1_,
														 *nu_, 
														       *alpha1_,*alpha2_,
														       devHIntegCache_, 
														 thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    _delInfectionTimeIntegral<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(ii, 
														 thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														 notification,
														 *devC_,
														 devEventTimes_, 
														 eventTimesPitch_, 
														 devSusceptibility_,
														 *delta_, 
														 *omega_, 
														 *beta2_,
														 *nu_, 
														  *alpha1_,*alpha2_, 
														  devHIntegCache_,
														 thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();
    if(maxBpG > 1) {
      CUDPPResult res = cudppReduce(addReduce_, &devComponents_->integral,
				    thrust::raw_pointer_cast(&(*devWorkspace_)[0]), maxBpG);
      if (res != CUDPP_SUCCESS)
	throw std::runtime_error(
				 "cudppReduce failed in GpuLikelihood::UpdateInfectionTime()");
    }
    else {
      checkCudaError(cudaMemcpy(&devComponents_->integral, thrust::raw_pointer_cast(&(*devWorkspace_)[0]), sizeof(float), cudaMemcpyDeviceToDevice));
#ifndef NDEBUG
      cerr << __FUNCTION__ << ": blocksPerGrid = " << blocksPerGridD << endl;
#endif
    }
    _delInfectionTimeProduct<DistanceKernel,true><<<blocksPerGridD, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(ii, 
														thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														notification,
														*devD_, 
														devEventTimes_, 
														eventTimesPitch_,
														devSusceptibility_, 
														*gamma1_, 
														*delta_, 
														*omega_,
														*beta1_,
														*nu_, 
														      *alpha1_,*alpha2_, 
														thrust::raw_pointer_cast(&(*devProduct_)[0]));
    _delInfectionTimeProduct<Identity,false><<<blocksPerGridC, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(ii, 
														thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), 
														notification,
														*devC_, 
														devEventTimes_, 
														eventTimesPitch_,
														devSusceptibility_, 
														*gamma1_, 
														*delta_, 
														*omega_,
														*beta2_,
														*nu_, 
														 *alpha1_,*alpha2_, 
														thrust::raw_pointer_cast(&(*devProduct_)[0]));
    checkCudaError(cudaGetLastError());

    // Make the change to the population
    bool haveNewI1 = false;
    devInfecIdx_->erase(devInfecIdx_->begin() + ii);
    hostInfecIdx_->erase(hostInfecIdx_->begin() + ii);
    hostSuscOccults_->push_back(i);
    eventTimesPtr[i] = notification;
    (*devProduct_)[i] = 1.0f;

    if (i == I1Idx_)
      {
        UpdateI1();
        CalcBgIntegral();
        haveNewI1 = true;
#ifndef NDEBUG
        std::cerr << "New I1" << std::endl;
#endif
      }

    ReduceProductVector();

    // Collect results and update likelihood
    //LikelihoodComponents localUpdate;
    //checkCudaError(cudaMemcpy(&localUpdate, devComponents_, sizeof(LikelihoodComponents), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    hostComponents_->integral = savedIntegral
      + hostComponents_->integral * *gamma1_;
    if (!haveNewI1) {
      hostComponents_->bgIntegral += *epsilon1_ * (notification - oldI);
    }

    logLikelihood_ = hostComponents_->logProduct
      - (hostComponents_->integral + hostComponents_->bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
	      << timeinseconds(start, end) << std::endl;
    std::cerr.precision(20);
    std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
	      << std::endl;
#endif

#ifndef NDEBUG
    cerr << __FUNCTION__ << " (likelihood)\n";
    PrintLikelihoodComponents();
    PrintParameters();
    PrintEventTimes();
    cerr << endl;
#endif

  }

  float
  GpuLikelihood::GetIN(const size_t index)
  {
    int i = (*hostInfecIdx_)[index].ptr;
    //thrust::device_vector<float> res(1);
    thrust::device_ptr<float> et(devEventTimes_);
    //thrust::transform(et + eventTimesPitch_ + i, et + eventTimesPitch_ + i + 1,
    //    et + i, &res[0], thrust::minus<float>());

    return et[eventTimesPitch_+i] - et[i];

    //return res[0];
  }

  float
  GpuLikelihood::GetLogLikelihood() const
  {
    return logLikelihood_;
  }

  float
  GpuLikelihood::GetN(const int idx) const
  {
    float rv;
    checkCudaError(
		   cudaMemcpy(devEventTimes_+idx+eventTimesPitch_,&rv,sizeof(float), cudaMemcpyDeviceToHost));
    return rv;
  }

  void
  GpuLikelihood::LazyAddInfecTime(const int idx, const float inTime)
  {
    thrust::device_ptr<float> eventTimePtr(devEventTimes_);
    eventTimePtr[idx] = eventTimePtr[idx + eventTimesPitch_] - inTime;
    devInfecIdx_->push_back(idx);
    devProduct_->push_back(0.0f);
    cudaDeviceSynchronize();
  }

  void
  GpuLikelihood::GetSumInfectivityPow(float* result) const
  {
    for (size_t k = 0; k < numSpecies_; ++k)
      {
        result[k] = indirectedSum(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
				  numKnownInfecs_, devAnimalsInfPow_ + animalsInfPowPitch_ * k);
      }
  }

  void
  GpuLikelihood::GetSumSusceptibilityPow(float* result) const
  {
    for (size_t k = 0; k < numSpecies_; ++k)
      {
        thrust::device_ptr<float> p(
				    devAnimalsSuscPow_ + animalsSuscPowPitch_ * k);
        result[k] = thrust::reduce(p, p + popSize_);
      }
  }

  float
  GpuLikelihood::GetMeanI2N() const
  {
    thrust::device_ptr<float> p(devEventTimes_);
    thrust::device_vector<float> buff(numKnownInfecs_);
    thrust::transform(p + eventTimesPitch_,
		      p + eventTimesPitch_ + numKnownInfecs_, p, buff.begin(),
		      thrust::minus<float>());
    return thrust::reduce(buff.begin(), buff.end()) / (float) numKnownInfecs_;
  }

  float
  GpuLikelihood::GetMeanOccI() const
  {
    size_t numOccs = GetNumOccults();
    if (numOccs == 0)
      return 0.0f;
    else
      {
        float sumI = indirectedSum(
				   thrust::raw_pointer_cast(&(*devInfecIdx_)[numKnownInfecs_]),
				   GetNumOccults(), devEventTimes_);
        float sumN = indirectedSum(
				   thrust::raw_pointer_cast(&(*devInfecIdx_)[numKnownInfecs_]),
				   GetNumOccults(), devEventTimes_ + eventTimesPitch_);
        return (sumN - sumI) / GetNumOccults();
      }
  }

  float
  GpuLikelihood::NonCentreInfecTimes(const float oldGamma, const float newGamma,
				     const float prob)
  {

    // Generate random numbers
    thrust::device_vector<float> seeds(GetNumKnownInfecs());
    curandStatus_t status = curandGenerateUniform(cuRand_,
						  thrust::raw_pointer_cast(&seeds[0]), GetNumKnownInfecs());
    if (status != CURAND_STATUS_SUCCESS)
      {
        throw std::runtime_error("curandGenerateUniform failed");
      }

    float logLikDiff = 0.0f;

    int dimGrid((GetNumKnownInfecs() + THREADSPERBLOCK - 1) / THREADSPERBLOCK);

    // Update the infection times
    _nonCentreInfecTimes<<<dimGrid, THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]), GetNumKnownInfecs(), devEventTimes_, eventTimesPitch_, oldGamma/newGamma, thrust::raw_pointer_cast(&seeds[0]), prob);

    // Do known bit -- GPU in parallel with CPU
    _knownInfectionsLikelihoodPNC<<<dimGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(raw_pointer_cast(&(*devInfecIdx_)[0]), GetNumKnownInfecs(), devEventTimes_, eventTimesPitch_, *a_, oldGamma, newGamma,
											       thrust::raw_pointer_cast(&seeds[0]), prob, thrust::raw_pointer_cast(&(*devWorkspace_)[0]));
    checkCudaError(cudaGetLastError());

    // Todo: GPU-ise this!
    for (size_t i = GetNumKnownInfecs(); i < GetNumInfecs(); ++i)
      {
        float Ii, Ni;
        checkCudaError(
		       cudaMemcpyAsync(&Ii, devEventTimes_+(*hostInfecIdx_)[i].ptr, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaError(
		       cudaMemcpyAsync(&Ni, devEventTimes_+eventTimesPitch_+(*hostInfecIdx_)[i].ptr, sizeof(float), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        logLikDiff += logf(gsl_cdf_gamma_Q(Ni - Ii, *a_, 1.0 / newGamma))
	  - logf(gsl_cdf_gamma_Q(Ni - Ii, *a_, 1.0 / oldGamma));
      }

    logLikDiff += thrust::reduce(devWorkspace_->begin(),
				 devWorkspace_->begin() + dimGrid);


    return logLikDiff;
  }

  void
  GpuLikelihood::GetInfectiousPeriods(std::vector<EpiRisk::IPTuple_t>& periods)
  {
    periods.resize(GetNumInfecs());

    thrust::device_vector<float> devOutputVec(GetNumInfecs());
    int blocksPerGrid((GetNumInfecs() + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
    _collectInfectiousPeriods<<<blocksPerGrid, THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&(*devInfecIdx_)[0]),
								  GetNumInfecs(),
								  devEventTimes_,
								  eventTimesPitch_,
								  thrust::raw_pointer_cast(&devOutputVec[0]));

    thrust::host_vector<float> outputVec(GetNumInfecs());
    outputVec = devOutputVec;
    for (size_t i = 0; i < GetNumInfecs(); ++i)
      {
        periods[i].idx = (*hostInfecIdx_)[i].ptr;
        periods[i].val = outputVec[i];
      }
  }

  void
  GpuLikelihood::GetInfectionTimes(std::vector<EpiRisk::IPTuple_t>& times)
  {
    times.resize(GetNumInfecs());

    std::vector<float> buff(popSize_);

    checkCudaError(cudaMemcpy(buff.data(), devEventTimes_, sizeof(float)*popSize_, cudaMemcpyDeviceToHost));

    for(size_t i=0; i<GetNumInfecs(); ++i)
      {
	times[i].idx = (*hostInfecIdx_)[i].ptr;
	times[i].val = buff[times[i].idx];
      }
  }

  std::ostream&
  operator <<(std::ostream& out, const GpuLikelihood& likelihood)
  {

    thrust::device_vector<float> devOutputVec(likelihood.GetNumInfecs());
    int blocksPerGrid(
		      (likelihood.GetNumInfecs() + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
    _collectInfectiousPeriods<<<blocksPerGrid, THREADSPERBLOCK>>>(thrust::raw_pointer_cast(&likelihood.devInfecIdx_->operator[](0)),
								  likelihood.GetNumInfecs(),
								  likelihood.devEventTimes_,
								  likelihood.eventTimesPitch_,
								  thrust::raw_pointer_cast(&devOutputVec[0]));

    thrust::host_vector<float> outputVec(likelihood.GetNumInfecs());
    outputVec = devOutputVec;

    out << likelihood.hostPopulation_[likelihood.hostInfecIdx_->operator[](0).ptr].id << ":"
        << outputVec[0];
    for (size_t i = 1; i < likelihood.GetNumInfecs(); ++i)
      out << ","
          << likelihood.hostPopulation_[likelihood.hostInfecIdx_->operator[](i).ptr].id
          << ":" << outputVec[i];

    return out;
  }

  void
  GpuLikelihood::RefreshParameters()
  {
    float tmp[TICKLEVELS];
    
    for(int p=0; p<TICKLEVELS; ++p)
      tmp[p] = phi_[p];
    checkCudaError(
		   cudaMemcpy(devPhi_, tmp, TICKLEVELS*sizeof(float), cudaMemcpyHostToDevice)
		   );
    
    CalcHFuncIntegCache(*alpha1_, *alpha2_, devHIntegCache_,true);
  }

  void
  GpuLikelihood::PrintLikelihoodComponents() const
  {
    cudaDeviceSynchronize();
    cerr.precision(15);
    cerr << "Background: " << hostComponents_->bgIntegral << "\n";
    cerr << "Integral: " << hostComponents_->integral << "\n";
    cerr << "Product: " << hostComponents_->logProduct << "\n";
  }

  void GpuLikelihood::PrintParameters() const
  {
    cerr << "Epsilon1: " << *epsilon1_ << "\n";
    cerr << "Gamma1: " << *gamma1_ << "\n";
    cerr << "Delta: " << *delta_ << "\n";
    cerr << "Omega: " << *omega_ << "\n";
    cerr << "beta1:" << *beta1_ << "\n";
    cerr << "beta2:" << *beta2_ << "\n";
    cerr << "alpha1: " << *alpha1_ << "\n";
    cerr << "alpha2: " << *alpha2_ << "\n";
    cerr << "a: " << *a_ << "\n";
    cerr << "b: " << *b_ << endl;
    cerr << "ObsTime: " << obsTime_ << "\n";
    cerr << "I1Idx = " << I1Idx_ << "\n";
    cerr << "I1Time = " << I1Time_ << "\n";
  }

  void
  GpuLikelihood::PrintEventTimes() const
  {
   
    // Obtain event times from GPU
    cudaDeviceSynchronize();
    float *events = new float[numKnownInfecs_*NUMEVENTS];
    checkCudaError(cudaMemcpy2D(events, numKnownInfecs_*sizeof(float), devEventTimes_, eventTimesPitch_*sizeof(float), numKnownInfecs_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    std::vector<std::string> ids;
    GetIds(ids);
    cerr << "===EVENTS===\n";
    for(int i = 0; i<numKnownInfecs_; ++i)
      cerr << ids[i] << "\t" << events[i] << "\t" << events[i+numKnownInfecs_] << "\t" << events[i+2*numKnownInfecs_] << "\n";
    cerr << "============" << endl;

    delete[] events;
  }
  
  void
  GpuLikelihood::PrintDistMatrix() const
  {
    cerr << "======DIST MATRIX======";

    // Copy distance matrix to host
    CsrMatrix *myCSR = new CsrMatrix;
    *myCSR = *devD_;

    myCSR->rowPtr = new int[myCSR->n];
    myCSR->colInd = new int[myCSR->nnz];
    myCSR->val = new float[myCSR->nnz];

    checkCudaError(cudaMemcpy(myCSR->rowPtr, devD_->rowPtr, myCSR->n*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(myCSR->colInd, devD_->colInd, myCSR->nnz*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(myCSR->val, devD_->val, myCSR->nnz*sizeof(int), cudaMemcpyDeviceToHost));

    for(int i=0; i<numKnownInfecs_; ++i) {
      for(int j=0; j<numKnownInfecs_; ++j) {
	cerr << GetDistElement(myCSR, i, j) << "\t";
      }
      cerr << "\n";
    }

    cerr << "=======================";

    delete[] myCSR->rowPtr;
    delete[] myCSR->colInd;
    delete[] myCSR->val;
    delete myCSR;
  }

  void
  GpuLikelihood::CheckSuscep() const
  {
    float* tmp = new float[popSize_];
    checkCudaError(cudaMemcpy(tmp, devSusceptibility_, popSize_*sizeof(float), cudaMemcpyDeviceToHost));
    float sum=0.0;
    for(int i=0; i<popSize_; ++i) {
      cerr << hostPopulation_[i].id << ": " << tmp[i] << "\n";
      sum += tmp[i];
    }
    cerr << "Sum susceptibility = " << sum << endl;

    delete[] tmp;
  }

  void
  GpuLikelihood::PrintProdVector() const
  {
    cerr << "======PROD VECTOR=======\n";
    thrust::host_vector<float> prod = GetProdVector();
    for(int i = 0; i<hostPopulation_.size(); ++i)
      cerr << hostPopulation_[i].id << "\t" << prod[i] << "\n";
  }

  void
  GpuLikelihood::DumpAnimals() const
  {
    float* tmp = new float[popSize_];
    checkCudaError(cudaMemcpy(tmp, devSusceptibility_, sizeof(float)*popSize_, cudaMemcpyDeviceToHost));
    float tmpPhi[TICKLEVELS];
    checkCudaError(cudaMemcpy(tmpPhi, devPhi_, sizeof(float)*TICKLEVELS, cudaMemcpyDeviceToHost));
    int idx = 0;
    while(hostPopulation_[idx].id != "HU-7395-0134") idx++;
    cerr << hostPopulation_[idx].id << "\t" << tmp[idx] << ", phi[2]=" << phi_[2] <<", devphi[2]="<<tmpPhi[2]<<"\n";

    delete[] tmp;
  }


} // namespace EpiRisk



