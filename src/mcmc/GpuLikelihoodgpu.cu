/*
 * GpuLikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <math_functions.h>
#include <device_functions.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/find.h>

#include "GpuLikelihood.hpp"

// Constants
const float UNITY = 1.0;
const float ZERO = 0.0;

inline
double
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

#define checkCudaError(err)  __checkCudaError(err, __FILE__, __LINE__)

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

template<typename T1, typename T2>
struct IndirectMin
{
  __host__ __device__
  IndirectMin(T2* ptr) : ptr_(ptr) {};

  __host__ __device__
  bool
  operator()(const T1 lhs, const T1 rhs) const
  {
    return ptr_[lhs] < ptr_[rhs];
  }
private:
  T2* ptr_;
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

__device__ void
_shmemReduce(float* buff)
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
      volatile float* vbuff = buff;
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 32];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 16];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 8];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 4];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 2];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 1];
    }

  __syncthreads();
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
      N = fminf(N,    R);
      I = fminf(I,    N);

      data[tid + pitch * 2] = R;
      data[tid + pitch] = N;
      data[tid] = I;
    }
}

__global__ void
_calcIntegral(const unsigned int* infecIdx, const int infecSize, int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    float* output)
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
      int i = infecIdx[row];

      int begin = DRowPtr[i];
      int end = DRowPtr[i + 1];
      float Ii = eventTimes[i];
      float Ni = eventTimes[i + eventTimesPitch];
      float Ri = eventTimes[i + eventTimesPitch * 2];

      float threadSum = 0.0f;
      for (int jj = begin + lane; jj < end; jj += 32)
        {
          // Integrated infection pressure
          float Ij = eventTimes[DColInd[jj]];
          float betaij = fminf(Ni, Ij) - fminf(Ii, Ij);
          betaij += gamma2 * (fminf(Ri, Ij) - fminf(Ni, Ij));

          // Apply distance kernel and suscep
          betaij *= delta / (delta * delta + D[jj]);
          betaij *= susceptibility[DColInd[jj]];
          threadSum += betaij;
        }
      buff[threadIdx.x] = threadSum * infectivity[i];
    }

  // Reduce all warp sums and write to global memory.

  _shmemReduce(buff);

  if (threadIdx.x == 0)
    {
      output[blockIdx.x] = buff[0];
    }
}

__global__ void
_calcProduct(const unsigned int* infecIdx, const int infecSize, const int* DRowPtr, const int* DColInd,
    float* D, const float* eventTimes, const int eventTimesPitch,
    const float* susceptibility, const float* infectivity, const float epsilon,
    const float gamma1, const float gamma2, const float delta, float* prodCache)
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
      int j = infecIdx[row];

      int begin = DRowPtr[j];
      int end = DRowPtr[j + 1];

      float Ij = eventTimes[j];

      for (int ii = begin + lane; ii < end/* and DColInd[ii] < infecSize*/;
          ii += 32)
        {
          int i = DColInd[ii];
          float Ii = eventTimes[i];
          float Ni = eventTimes[eventTimesPitch + i];
          float Ri = eventTimes[eventTimesPitch * 2 + i];

          if(Ii < Ni) {
              float idxOnj = 0.0f;
              if (Ii < Ij and Ij <= Ni)
                idxOnj += 1.0f;
              else if (Ni < Ij and Ij <= Ri)
                idxOnj += gamma2;
              threadProdCache[threadIdx.x] += idxOnj * infectivity[i] * delta / (delta*delta + D[ii]);
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
        prodCache[j] = threadProdCache[threadIdx.x] * susceptibility[j]
            * gamma1 + epsilon;
    }
}

__global__ void
calcSpecPow(const unsigned int size, const int nSpecies, float* specpow,
    const int specpowPitch, const float* animals, const int animalsPitch,
    const float* powers)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size)
    {
      for (unsigned int col = 0; col < nSpecies; ++col)
        {
          specpow[col * specpowPitch + row] = powf(
              animals[col * animalsPitch + row], powers[col]);
        }
    }
}

__global__ void
_updateInfectionTimeIntegral(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    int* DRowPtr, int* DColInd, float* D, float* eventTimes,
    const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    float* output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__
  float buff[];
  buff[threadIdx.x] = 0.0f;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin)
    {
      int j = DColInd[begin + tid];

      float Ii = eventTimes[i];
      float Ni = eventTimes[i + eventTimesPitch];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];
      float Rj = eventTimes[j + eventTimesPitch * 2];

      float jOnIdx = 0.0f;
      if(Ij < Nj)
        {
          // Recalculate pressure from j on idx
          jOnIdx = (fminf(Nj, newTime) - fminf(Ij, newTime))
                      + gamma2 * (fminf(Rj, newTime) - fminf(Nj, newTime)); // New pressure
          jOnIdx -= (fminf(Nj, Ii) - fminf(Ii, Ij))
                  + gamma2 * (fminf(Rj, Ii) - fminf(Nj, Ii)); // Old pressure
          // Apply infec and suscep
          jOnIdx *= susceptibility[i];
          jOnIdx *= infectivity[j];
     }

      // Recalculate pressure from idx on j
      float IdxOnj = fminf(Ni, Ij) - fminf(newTime, Ij);
      IdxOnj -= fminf(Ni, Ij) - fminf(Ii, Ij);
      IdxOnj *= susceptibility[j];
      IdxOnj *= infectivity[i];

      buff[threadIdx.x] = (IdxOnj + jOnIdx) * (delta / (delta * delta + D[begin + tid]));

      // Reduce buffer into output
      _shmemReduce(buff);

    }

  if (threadIdx.x == 0)
    {
      output[blockIdx.x] = buff[0];
    }
}

__global__ void
_updateInfectionTimeProduct(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float epsilon, const float gamma1,
    const float gamma2, const float delta, float* prodCache)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__
  float buff[];
  buff[threadIdx.x] = 0.0f;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin) // Massive amount of wasted time just here!
    {
      int j = DColInd[begin + tid];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];

      if(Ij < Nj) {

          float Ii = eventTimes[i];
          float Ni = eventTimes[i + eventTimesPitch];

          float Rj = eventTimes[j + eventTimesPitch * 2];

          // Adjust product cache from idx on others
          float idxOnj = 0.0f;
          if (Ii < Ij and Ij <= Ni)
            idxOnj -= 1.0f;
          if (newTime < Ij and Ij <= Ni)
            idxOnj += 1.0f;

          idxOnj *= gamma1 * infectivity[i] * susceptibility[j] * delta
              / (delta * delta + D[begin + tid]);
          prodCache[j] += idxOnj;

          // Recalculate instantaneous pressure on idx
          float jOnIdx = 0.0f;
          if (Ij < newTime and newTime <= Nj)
            jOnIdx = 1.0f;
          else if (Nj < newTime and newTime <= Rj)
            jOnIdx = gamma2;

          jOnIdx *= susceptibility[i] * infectivity[j] * delta
                  / (delta * delta + D[begin + tid]);

          buff[threadIdx.x] = jOnIdx * gamma1;

          }

      _shmemReduce(buff);

      if (threadIdx.x == 0)
        _atomicAdd(prodCache + i, buff[0]); // Maybe better to create an external reduction buffer here.
      if (tid == 0)
        _atomicAdd(prodCache + i, epsilon);
    }
}



__global__ void
_addInfectionTimeIntegral(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    const int* DRowPtr, const int* DColInd, const float* D, const float* eventTimes,
    const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    float* output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__
  float buff[];
  buff[threadIdx.x] = 0.0f;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin)
    {
      int j = DColInd[begin + tid];

      float Ii = eventTimes[i];
      float Ni = eventTimes[i + eventTimesPitch];
      float Ri = eventTimes[i + eventTimesPitch * 2];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];
      float Rj = eventTimes[j + eventTimesPitch * 2];

      float jOnIdx = 0.0f;
      if(Ij < Nj)
        {
          // Calculate pressure from j on idx
          jOnIdx -= fminf(Nj, Ii) - fminf(Ij, Ii);
          jOnIdx -= gamma2 * (fminf(Rj, Ii) - fminf(Nj, Ii));
          jOnIdx += fminf(Nj, newTime) - fminf(Ij, newTime);
          jOnIdx += gamma2 * (fminf(Rj, newTime) - fminf(Nj, newTime));

          // Apply infec and suscep
          jOnIdx *= susceptibility[i];
          jOnIdx *= infectivity[j];
     }

      // Add pressure from idx on j
      float IdxOnj = fminf(Ni, Ij) - fminf(newTime, Ij);
      IdxOnj += gamma2 * (fminf(Ri, Ij) - fminf(Ni, Ij));
      IdxOnj *= susceptibility[j];
      IdxOnj *= infectivity[i];

      buff[threadIdx.x] = (IdxOnj + jOnIdx) * (delta / (delta * delta + D[begin + tid]));

      // Reduce buffer into output
      _shmemReduce(buff);
    }

  if (threadIdx.x == 0)
    {
      output[blockIdx.x] = buff[0];
    }
}



__global__ void
_delInfectionTimeIntegral(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    int* DRowPtr, int* DColInd, float* D, float* eventTimes,
    const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    float* output)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__
  float buff[];
  buff[threadIdx.x] = 0.0f;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin)
    {
      int j = DColInd[begin + tid];

      float Ii = eventTimes[i];
      float Ni = eventTimes[i + eventTimesPitch];
      float Ri = eventTimes[i + eventTimesPitch*2];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];
      float Rj = eventTimes[j + eventTimesPitch * 2];

      float jOnIdx = 0.0f;
      if(Ij < Nj)
        {
          // Recalculate pressure from j on idx
          jOnIdx -= fminf(Nj, Ii) - fminf(Ii, Ij) + gamma2 * (fminf(Rj, Ii) - fminf(Nj, Ii)); // Old pressure
          jOnIdx += fminf(Nj, Ni) - fminf(Ij, Ni) + gamma2 * (fminf(Rj, Ni) - fminf(Nj, Ni)); // New pressure
          // Apply infec and suscep
          jOnIdx *= susceptibility[i];
          jOnIdx *= infectivity[j];
     }

      // Subtract pressure from idx on j
      float IdxOnj = 0.0f;
      IdxOnj -= fminf(Ni, Ij) - fminf(Ii, Ij);
      IdxOnj -= gamma2 * (fminf(Ri, Ij) - fminf(Ni, Ij));
      IdxOnj *= susceptibility[j];
      IdxOnj *= infectivity[i];

      buff[threadIdx.x] = (IdxOnj + jOnIdx) * (delta / (delta * delta + D[begin + tid]));

      // Reduce buffer into output
      _shmemReduce(buff);

    }

  if (threadIdx.x == 0)
    {
      output[blockIdx.x] = buff[0];
    }
}


__global__ void
_addInfectionTimeProduct(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    const int* DRowPtr, const int* DColInd, const float* D,
    const float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float epsilon, const float gamma1,
    const float gamma2, const float delta, float* prodCache)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__
  float buff[];
  buff[threadIdx.x] = 0.0f;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin) // Massive amount of wasted time just here!
    {
      int j = DColInd[begin + tid];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];

      if(Ij < Nj) { // Only look at infected individuals

          float Ni = eventTimes[i + eventTimesPitch    ];
          float Ri = eventTimes[i + eventTimesPitch * 2];
          float Rj = eventTimes[j + eventTimesPitch * 2];

          // Adjust product cache from idx on others
          float idxOnj = 0.0f;
          if (newTime < Ij and Ij <= Ni)
            idxOnj += 1.0f;
          else if (Ni < Ij and Ij <= Ri)
            idxOnj += gamma2;

          idxOnj *= gamma1 * infectivity[i] * susceptibility[j] * delta
              / (delta * delta + D[begin + tid]);
          prodCache[j] += idxOnj;

          // Calculate instantaneous pressure on idx
          float jOnIdx = 0.0f;
          if (Ij < newTime and newTime <= Nj)
            jOnIdx = 1.0f;
          else if (Nj < newTime and newTime <= Rj)
            jOnIdx = gamma2;

          jOnIdx *= gamma1 * infectivity[j] * susceptibility[i] * delta
                  / (delta * delta + D[begin + tid]);

          buff[threadIdx.x] = jOnIdx;

          }

      _shmemReduce(buff);

      if (threadIdx.x == 0)
        _atomicAdd(prodCache + i, buff[0]);
      if (tid == 0)
        _atomicAdd(prodCache + i, epsilon);
    }
}


__global__ void
_delInfectionTimeProduct(const unsigned int idx, const unsigned int* infecIdx, const float newTime,
    int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float epsilon, const float gamma1,
    const float gamma2, const float delta, float* prodCache)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  int i = infecIdx[idx];
  int begin = DRowPtr[i];
  int end = DRowPtr[i + 1];

  if (tid < end - begin) // Massive amount of wasted time just here!
    {
      int j = DColInd[begin + tid];

      float Ij = eventTimes[j];
      float Nj = eventTimes[j + eventTimesPitch];

      if(Ij < Nj) {

          float Ii = eventTimes[i];
          float Ni = eventTimes[i + eventTimesPitch];
          float Ri = eventTimes[i + eventTimesPitch*2];

          // Adjust product cache from idx on others
          float idxOnj = 0.0;
          if (Ii < Ij and Ij <= Ni)
            idxOnj -= 1.0;
          else if(Ni < Ij and Ij <= Ri)
            idxOnj -= gamma2;

          idxOnj *= gamma1 * infectivity[i] * susceptibility[j] * delta
              / (delta * delta + D[begin + tid]);
          prodCache[j] += idxOnj;
          }
    }
}





GpuLikelihood::GpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic, DistMatrixImporter& distMatrix,
    const size_t nSpecies, const float obsTime, const bool occultsOnlyDC) :
    popSize_(0), numSpecies_(
        nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(0), sumI_(0), bgIntegral_(
        0.0), covariateCopies_(0), devAnimals_(NULL), animalsPitch_(0), devAnimalsInfPow_(
        NULL), devAnimalsSuscPow_(NULL), devEventTimes_(NULL), devSusceptibility_(
        NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(NULL), devDColInd_(
        NULL), epsilon_(0.0f), gamma1_(0.0f), gamma2_(0.0f), devXi_(
        NULL), devPsi_(NULL), devZeta_(NULL), devPhi_(NULL), delta_(0.0f)
{

  // Load data into host memory
  LoadPopulation(population);
  LoadEpidemic(epidemic);
  SortPopulation();
  LoadDistanceMatrix(distMatrix);

  // Set up on GPU
  SetSpecies();
  SetEvents();

  // Set up reference counter to covariate data
  covariateCopies_ = new size_t;
  *covariateCopies_ = 1;

  // Allocate product cache
  devProduct_.resize(maxInfecs_);
  thrust::fill(devProduct_.begin(), devProduct_.end(), 1.0f);

  // Allocate integral array
  int numRequiredThreads = maxInfecs_ * 32; // One warp per infection
  integralBuffSize_ = (numRequiredThreads + THREADSPERBLOCK - 1)
      / THREADSPERBLOCK;
  devIntegral_.resize(integralBuffSize_);

  // Parameters
  checkCudaError(cudaMalloc(&devXi_, numSpecies_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devPsi_, numSpecies_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devZeta_, numSpecies_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devPhi_, numSpecies_ * sizeof(float)));

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

}

// Copy constructor
GpuLikelihood::GpuLikelihood(const GpuLikelihood& other) :
   popSize_(other.popSize_), numKnownInfecs_(
        other.numKnownInfecs_), numSpecies_(other.numSpecies_), obsTime_(
        other.obsTime_), I1Time_(other.I1Time_), I1Idx_(other.I1Idx_), sumI_(
        other.sumI_), bgIntegral_(other.bgIntegral_), lp_(other.lp_), covariateCopies_(
        other.covariateCopies_), devAnimals_(other.devAnimals_), animalsPitch_(
        other.animalsPitch_), devDVal_(other.devDVal_), devDRowPtr_(
        other.devDRowPtr_), devDColInd_(other.devDColInd_), hostDRowPtr_(other.hostDRowPtr_),
        dnnz_(other.dnnz_), integralBuffSize_(
        other.integralBuffSize_), epsilon_(other.epsilon_), gamma1_(
        other.gamma1_), gamma2_(other.gamma2_), delta_(other.delta_)
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

  // Infection index
  devInfecIdx_ = other.devInfecIdx_;
  hostInfecIdx_ = other.hostInfecIdx_;

  // Allocate and copy product vector
  devProduct_ = other.devProduct_;
  devIntegral_ = other.devIntegral_;


  // Parameters -- Allocate and Copy
  checkCudaError(cudaMalloc(&devXi_, numSpecies_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devXi_, other.devXi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devPsi_, numSpecies_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devPsi_, other.devPsi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devZeta_, numSpecies_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devZeta_, other.devZeta_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devPhi_, numSpecies_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devPhi_, other.devPhi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));

  // BLAS handles
  cudaBLAS_ = other.cudaBLAS_;
  cudaSparse_ = other.cudaSparse_;
  crsDescr_ = other.crsDescr_;

  ++*covariateCopies_; // Increment copies of covariate data

  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;

}

// Assignment constructor
const GpuLikelihood&
GpuLikelihood::operator=(const GpuLikelihood& other)
{
  timeval start, end;
  gettimeofday(&start, NULL);
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
  devInfecIdx_ = other.devInfecIdx_;
  hostInfecIdx_ = other.hostInfecIdx_;

  // copy product vector
  devProduct_ = other.devProduct_;
  devIntegral_ = other.devIntegral_;

  // Host Parameters Copy
  epsilon_ = other.epsilon_;
  gamma1_ = other.gamma1_;
  gamma2_ = other.gamma2_;
  delta_ = other.delta_;
  xi_ = other.xi_;
  psi_ = other.psi_;
  zeta_ = other.zeta_;
  phi_ = other.phi_;

  RefreshParameters();

  // Likelihood components
  integral_ = other.integral_;
  bgIntegral_ = other.bgIntegral_;
  lp_ = other.lp_;

  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;

  return *this;
}

GpuLikelihood::~GpuLikelihood()
{
  if (*covariateCopies_ == 1) // We're the last copy to be destroyed
    {
      cudaFree(devAnimals_);
      cudaFree(devDVal_);
      cudaFree(devDRowPtr_);
      cudaFree(devDColInd_);
      delete[] hostDRowPtr_;
      cublasDestroy(cudaBLAS_);
      cusparseDestroy(cudaSparse_);

      delete covariateCopies_;
    }

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

  if (devXi_)
    cudaFree(devXi_);
  if (devPsi_)
    cudaFree(devPsi_);
  if (devZeta_)
    cudaFree(devZeta_);
  if (devPhi_)
    cudaFree(devPhi_);

}

void
GpuLikelihood::SetEvents()
{

  // Set up Species and events
  float* eventsMatrix = new float[popSize_*numSpecies_];
  Population::iterator it = hostPopulation_.begin();
  for(size_t i=0; i<popSize_; ++i)
    {
      eventsMatrix[i] = it->I;
      eventsMatrix[i+popSize_] = it->N;
      eventsMatrix[i+popSize_*2] = it->R;
      ++it;
    }

  // Allocate Event times - popSize_ * NUMEVENTS matrix
    checkCudaError(
        cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS));
    eventTimesPitch_ /= sizeof(float);


  // Get event times into GPU memory
  cudaError_t rv = cudaMemcpy2D(devEventTimes_,
      eventTimesPitch_ * sizeof(float), eventsMatrix, popSize_ * sizeof(float),
      popSize_ * sizeof(float), NUMEVENTS, cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copying event times to device failed", rv);

  // Set any event times greater than obsTime to obsTime
  int blocksPerGrid = (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  _sanitizeEventTimes<<<blocksPerGrid, THREADSPERBLOCK>>>(devEventTimes_, eventTimesPitch_, obsTime_, popSize_);
  checkCudaError(cudaGetLastError());

  thrust::device_ptr<float> p(devEventTimes_);
  hostInfecIdx_.clear();
  for(size_t i=0; i<numKnownInfecs_; ++i)
    {
      hostInfecIdx_.push_back(i);
    }
  devInfecIdx_ = hostInfecIdx_;
  cudaDeviceSynchronize();
}

void
GpuLikelihood::SetSpecies()
{

  // Set up Species and events
  float* speciesMatrix = new float[popSize_ * numSpecies_];
  Population::const_iterator it = hostPopulation_.begin();
  for(size_t i=0; i<hostPopulation_.size(); ++i)
    {
      speciesMatrix[i] = it->cattle;
      speciesMatrix[i+hostPopulation_.size()] = it->pigs;
      speciesMatrix[i+hostPopulation_.size()*2] = it->sheep;
      ++it;
    }


  // Allocate Animals_
  checkCudaError(
      cudaMallocPitch(&devAnimals_, &animalsPitch_, popSize_ * sizeof(float), numSpecies_));
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

  cudaError_t rv = cudaMemcpy2D(devAnimals_, animalsPitch_ * sizeof(float),
      speciesMatrix, popSize_ * sizeof(float), popSize_ * sizeof(float), numSpecies_,
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Failed copying species data to device", rv);

}

void
GpuLikelihood::SetDistance(const float* data, const int* rowptr,
    const int* colind)
{

  checkCudaError(cudaMalloc(&devDVal_, dnnz_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devDRowPtr_, (popSize_ + 1) * sizeof(int)));
  checkCudaError(cudaMalloc(&devDColInd_, dnnz_ * sizeof(float)));
  hostDRowPtr_ = new int[popSize_ + 1];

  checkCudaError(
      cudaMemcpy(devDVal_, data, dnnz_ * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(devDRowPtr_, rowptr, (popSize_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(devDColInd_, colind, dnnz_ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(hostDRowPtr_, rowptr, (popSize_ + 1)*sizeof(int), cudaMemcpyHostToHost));
}

void
GpuLikelihood::RefreshParameters()
{

  float* tmp = new float[numSpecies_];


  for(size_t i=0; i<numSpecies_; ++i) tmp[i] = xi_[i];
  checkCudaError(
      cudaMemcpy(devXi_, tmp, numSpecies_ * sizeof(float), cudaMemcpyHostToDevice));

  for(size_t i=0; i<numSpecies_; ++i) tmp[i] = xi_[i];
  checkCudaError(
      cudaMemcpy(devPsi_, tmp, numSpecies_ * sizeof(float), cudaMemcpyHostToDevice));

  for(size_t i=0; i<numSpecies_; ++i) tmp[i] = xi_[i];
  checkCudaError(
      cudaMemcpy(devZeta_, tmp, numSpecies_ * sizeof(float), cudaMemcpyHostToDevice));

  for(size_t i=0; i<numSpecies_; ++i) tmp[i] = xi_[i];
  checkCudaError(
      cudaMemcpy(devPhi_, tmp, numSpecies_ * sizeof(float), cudaMemcpyHostToDevice));

  delete[] tmp;
}

inline
void
GpuLikelihood::CalcInfectivityPow()
{
  int dimBlock(THREADSPERBLOCK);
  int dimGrid((maxInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
calcSpecPow<<<dimGrid, dimBlock>>>(maxInfecs_,numSpecies_,devAnimalsInfPow_, animalsInfPowPitch_,devAnimals_,animalsPitch_,devPsi_);
                checkCudaError(cudaGetLastError());
}

inline
void
GpuLikelihood::CalcInfectivity()
{

  // Now calculate infectivity
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, maxInfecs_, numSpecies_,
      &UNITY, devAnimalsInfPow_, animalsInfPowPitch_, devXi_, 1, &ZERO,
      devInfectivity_, 1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in infectivity: " << blasStat_ << std::endl;
    }

}

inline
void
GpuLikelihood::CalcSusceptibilityPow()
{
  int dimBlock(THREADSPERBLOCK);
  int dimGrid((popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
calcSpecPow<<<dimGrid, dimBlock>>>(popSize_,numSpecies_,devAnimalsSuscPow_,animalsSuscPowPitch_, devAnimals_,animalsPitch_,devPhi_);
                checkCudaError(cudaGetLastError());
}

inline
void
GpuLikelihood::CalcSusceptibility()
{
  // Calculates susceptibility powers and sums over suscept.
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, popSize_, numSpecies_, &UNITY,
      devAnimalsSuscPow_, animalsSuscPowPitch_, devZeta_, 1, &ZERO,
      devSusceptibility_, 1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in susceptibility: " << blasStat_ << std::endl;
    }
}

inline
void
GpuLikelihood::UpdateI1()
{
  thrust::device_vector<unsigned int>::iterator myMin;
  myMin = thrust::min_element(devInfecIdx_.begin(), devInfecIdx_.end(), IndirectMin<unsigned int,float>(devEventTimes_));
  I1Idx_ = *myMin;

  thrust::device_ptr<float> v(devEventTimes_);
  I1Time_ = v[I1Idx_];
}
inline
void
GpuLikelihood::CalcBgIntegral()
{
  thrust::device_ptr<float> v(devEventTimes_);
  sumI_ = thrust::reduce(v, v + popSize_, 0.0f, thrust::plus<float>());

  bgIntegral_ = sumI_ - (v[I1Idx_]*popSize_);
  bgIntegral_ *= epsilon_;
}

inline
void
GpuLikelihood::CalcProduct()
{

  _calcProduct<<<integralBuffSize_,THREADSPERBLOCK>>>(devInfecIdx_.data().base(),devInfecIdx_.size(),devDRowPtr_,devDColInd_,devDVal_,
      devEventTimes_,eventTimesPitch_,devSusceptibility_,devInfectivity_,epsilon_,gamma1_,gamma2_,delta_,devProduct_.data().base());
  checkCudaError(cudaGetLastError());

  devProduct_[I1Idx_] = 1.0f;

  lp_ = thrust::transform_reduce(devProduct_.begin(), devProduct_.end(), Log<float>(),
      0.0f, thrust::plus<float>());
}

inline
void
GpuLikelihood::CalcIntegral()
{

  int numRequiredThreads = devInfecIdx_.size() * 32; // One warp per infection
  int integralBuffSize = (numRequiredThreads + THREADSPERBLOCK - 1)
       / THREADSPERBLOCK;


_calcIntegral<<<integralBuffSize_,THREADSPERBLOCK>>>(devInfecIdx_.data().base(),devInfecIdx_.size(),devDRowPtr_,devDColInd_,devDVal_,
      devEventTimes_,eventTimesPitch_,devSusceptibility_,devInfectivity_,gamma2_,delta_,devIntegral_.data().base());
        checkCudaError(cudaGetLastError());

  integral_ = thrust::reduce(devIntegral_.begin(), devIntegral_.begin() + integralBuffSize) * gamma1_;
}

void
GpuLikelihood::FullCalculate()
{

  timeval start, end;
  gettimeofday(&start, NULL);
  RefreshParameters();
  CalcInfectivityPow();
  CalcInfectivity();
  CalcSusceptibilityPow();
  CalcSusceptibility();

  UpdateI1();
  CalcProduct();
  CalcIntegral();
  CalcBgIntegral();

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
      << std::endl;

}

void
GpuLikelihood::Calculate()
{
  timeval start, end;
  gettimeofday(&start, NULL);
  RefreshParameters();
  CalcInfectivity();
  CalcSusceptibility();

  UpdateI1();
  CalcIntegral();
  CalcProduct();
  CalcBgIntegral();
  logLikelihood_ = lp_ - (integral_ + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr.precision(20);
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
      << std::endl;
}

void
GpuLikelihood::UpdateInfectionTime(const unsigned int idx, const float inTime)
{
  // Require to know number of cols per row -- probably store in host mem.
  // Also, may be optimal to use a much lower THREADSPERBLOCK than the app-wide setting.


  timeval start, end;
  gettimeofday(&start, NULL);

  thrust::device_ptr<float> eventTimesPtr(devEventTimes_);
  float newTime = *(eventTimesPtr+eventTimesPitch_+idx) - inTime;

  int blocksPerGrid = (hostDRowPtr_[idx + 1] - hostDRowPtr_[idx]
      + THREADSPERBLOCK - 1) / THREADSPERBLOCK + 1;
  _updateInfectionTimeIntegral<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, devInfecIdx_.data().base(), newTime,
      devDRowPtr_, devDColInd_, devDVal_,
      devEventTimes_, eventTimesPitch_, devSusceptibility_,
      devInfectivity_, gamma2_, delta_, devIntegral_.data().base());
      checkCudaError(cudaGetLastError());

  integral_ += thrust::reduce(devIntegral_.begin(), devIntegral_.begin() + blocksPerGrid) * gamma1_;


  // If a new I1 is created by moving a non-I1 infection time, zero out I1
  if (newTime < I1Time_ and idx != I1Idx_) devProduct_[I1Idx_] = epsilon_;

  devProduct_[idx] = 0.0f; // Zero out product entry for idx.
_updateInfectionTimeProduct<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, devInfecIdx_.data().base(), newTime, devDRowPtr_,
      devDColInd_, devDVal_, devEventTimes_, eventTimesPitch_,
      devSusceptibility_, devInfectivity_, epsilon_, gamma1_, gamma2_,
      delta_, devProduct_.data().base());
      checkCudaError(cudaGetLastError());

  // Make the change to the population
  eventTimesPtr[idx] = newTime;

  UpdateI1();
  CalcBgIntegral();

  devProduct_[I1Idx_] = 1.0f;
  lp_ = thrust::transform_reduce(devProduct_.begin(), devProduct_.end(), Log<float>(),
      0.0f, thrust::plus<float>());

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);

  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr.precision(20);
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
      << std::endl;
  std::cerr << "I1: " << I1Idx_ << " at " << I1Time_ << std::endl;
}


void
GpuLikelihood::AddInfectionTime(const unsigned int idx, const float inTime)
{
  // Require to know number of cols per row -- probably store in host mem.
  // Also, may be optimal to use a much lower THREADSPERBLOCK than the app-wide setting.


  timeval start, end;
  gettimeofday(&start, NULL);

  if(idx < numKnownInfecs_ or idx >= maxInfecs_) throw std::range_error("Invalid idx in GpuLikelihood::AddInfectionTime");

  thrust::device_ptr<float> eventTimesPtr(devEventTimes_);
  float newTime = *(eventTimesPtr+eventTimesPitch_+idx) - inTime;

  // Ready the product cache to receive pressure
  devInfecIdx_.push_back(idx);
  hostInfecIdx_.push_back(idx);

  unsigned int addIdx = devInfecIdx_.size()-1;


  int blocksPerGrid = (hostDRowPtr_[idx + 1] - hostDRowPtr_[idx]
      + THREADSPERBLOCK - 1) / THREADSPERBLOCK + 1;
  _addInfectionTimeIntegral<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, devInfecIdx_.data().base(), newTime,
      devDRowPtr_, devDColInd_, devDVal_,
      devEventTimes_, eventTimesPitch_, devSusceptibility_,
      devInfectivity_, gamma2_, delta_, devIntegral_.data().base());
      checkCudaError(cudaGetLastError());

  integral_ += thrust::reduce(devIntegral_.begin(), devIntegral_.begin() + blocksPerGrid) * gamma1_;


  // If a new I1 is created by moving a non-I1 infection time, set the old I1 to epsilon
  if (newTime < I1Time_) devProduct_[I1Idx_] = epsilon_;

  devProduct_[idx] = 0.0f;
_addInfectionTimeProduct<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(addIdx, thrust::raw_pointer_cast(&devInfecIdx_[0]), newTime, devDRowPtr_,
      devDColInd_, devDVal_, devEventTimes_, eventTimesPitch_,
      devSusceptibility_, devInfectivity_, epsilon_, gamma1_, gamma2_,
      delta_, thrust::raw_pointer_cast(&devProduct_[0]));
      checkCudaError(cudaGetLastError());

  // Make the change to the population
  eventTimesPtr[idx] = newTime;

  UpdateI1();
  CalcBgIntegral();

  // Reduce product vector, correcting for I1
  devProduct_[I1Idx_] = 1.0f;
  lp_ = thrust::transform_reduce(devProduct_.begin(), devProduct_.end(), Log<float>(),
      0.0f, thrust::plus<float>());


  logLikelihood_ = lp_ - (integral_ + bgIntegral_);



  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr.precision(20);
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
      << std::endl;
}


void
GpuLikelihood::DeleteInfectionTime(const unsigned int idx)
{
  // Require to know number of cols per row -- probably store in host mem.
  // Also, may be optimal to use a much lower THREADSPERBLOCK than the app-wide setting.


  timeval start, end;
  gettimeofday(&start, NULL);

  // Range check
  if(idx < numKnownInfecs_ or idx >= devInfecIdx_.size()) throw std::range_error("Invalid idx in GpuLikelihood::DeleteInfectionTime");

  thrust::device_ptr<float> eventTimesPtr(devEventTimes_);
  unsigned int i = hostInfecIdx_[idx];
  float notification = eventTimesPtr[i + eventTimesPitch_];
  devIntegral_.assign(devIntegral_.size(), 0.0f);

  int blocksPerGrid = (hostDRowPtr_[i + 1] - hostDRowPtr_[i]
      + THREADSPERBLOCK - 1) / THREADSPERBLOCK + 1;
  _delInfectionTimeIntegral<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, devInfecIdx_.data().base(), notification,
      devDRowPtr_, devDColInd_, devDVal_,
      devEventTimes_, eventTimesPitch_, devSusceptibility_,
      devInfectivity_, gamma2_, delta_, devIntegral_.data().base());
      checkCudaError(cudaGetLastError());

  integral_ += thrust::reduce(devIntegral_.begin(), devIntegral_.begin() + blocksPerGrid) * gamma1_;

_delInfectionTimeProduct<<<blocksPerGrid, THREADSPERBLOCK, THREADSPERBLOCK*sizeof(float)>>>(idx, devInfecIdx_.data().base(), notification, devDRowPtr_,
      devDColInd_, devDVal_, devEventTimes_, eventTimesPitch_,
      devSusceptibility_, devInfectivity_, epsilon_, gamma1_, gamma2_,
      delta_, devProduct_.data().base());
      checkCudaError(cudaGetLastError());

  // Make the change to the population
  eventTimesPtr[i] = notification;

  devProduct_[i] = 1.0f;
  devInfecIdx_.erase(devInfecIdx_.begin() + idx);
  hostInfecIdx_.erase(hostInfecIdx_.begin() + idx);

  UpdateI1();
  CalcBgIntegral();

  // Reduce product vector, correcting for I1
  devProduct_[I1Idx_] = 1.0f;
  lp_ = thrust::transform_reduce(devProduct_.begin(), devProduct_.end(), Log<float>(),
      0.0f, thrust::plus<float>());

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);


  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr.precision(20);
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
      << std::endl;
}

float
GpuLikelihood::GetIN(const size_t index)
{
  thrust::device_vector<float> res(1);
  thrust::device_ptr<float> et(devEventTimes_);
  thrust::transform(et + index + eventTimesPitch_, et + index + 1 + eventTimesPitch_, et + index, res, thrust::minus<float>());

  return *res;
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
  checkCudaError(cudaMemcpy(devEventTimes_+idx+eventTimesPitch_,&rv,sizeof(float), cudaMemcpyDeviceToHost));
  return rv;
}

void
GpuLikelihood::LazyAddInfecTime(const int idx, const float inTime)
{
  thrust::device_ptr<float> eventTimePtr(devEventTimes_);
  eventTimePtr[idx] = eventTimePtr[idx+eventTimesPitch_] - inTime;
  devInfecIdx_.push_back(idx);
  devProduct_.push_back(0.0f);
  cudaDeviceSynchronize();
}

