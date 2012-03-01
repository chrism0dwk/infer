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
#include <sys/time.h>

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
  struct Min
  {
    __device__ __host__
    T
    operator()(const T& a, const T& b) const
    {
      return a < b ? a : b;
    }
    __device__ __host__
    T
    initval() const
    {
      return HUGE_VAL;
    }
  };

template<typename T>
  struct Plus
  {
    __device__ __host__
    T
    operator()(const T& a, const T& b) const
    {
      return a + b;
    }
    __device__ __host__
    T
    initval() const
    {
      return 0;
    }
  };

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

__global__ void
calcDT(const float* D, const float* T, const int N, float* DT,
    const float delta)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    DT[idx] = delta / (delta * delta + D[idx]) * T[idx];
}

__global__ void
calcED(const float* eVal, const int* eRowPtr, const int* eColInd,
    const float* dVal, const int* dRowPtr, const int* dColInd, float* edVal,
    const int numInfecs, const float delta)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int wid = tid / 32; // Warp id
  int lane = tid & (32 - 1); // Id within a warp

  int row = wid;
  if (row < numInfecs)
    {
      int d_begin = dRowPtr[row]; //int d_end = dRowPtr[row+1];
      int e_begin = eRowPtr[row];
      int e_end = eRowPtr[row + 1];
      int rowLen = e_end - e_begin;

      for (int col = lane; col < rowLen; col += 32) // Loop over row in e
        {
          float d = dVal[d_begin + col];
          float e = eVal[e_begin + col];
          edVal[e_begin + col] = delta / (delta * delta + d) * e;

        }
    }
}

__global__ void
calcProdSusceptibility(const float* input, const float* infectivity,
    float* product, const int size, const int I1, const float epsilon,
    const float gamma1)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == I1)
    product[tid] = 1.0;
  else if (tid < size)
    {
      product[tid] = infectivity[tid] * input[tid] * gamma1 + epsilon;
    }
}

__global__ void
calcTold(const int infecSize, const int nnz, int* TRowPtr, int* TColInd,
    float* TVal, float* eventTimes, const int eventTimesPitch,
    const float gamma2, const float obsTime)
{
  // Each thread calculates a row i of the sparse matrix -- probably not efficient!

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < infecSize)
    {
      int begin = TRowPtr[i];
      int end = TRowPtr[i + 1];

      float Ii = eventTimes[i]; // First column  -- argument for row-major here, I would have thought.
      float Ni = eventTimes[eventTimesPitch + i]; // Second column
      float Ri = eventTimes[eventTimesPitch * 2 + i]; // Third column

      for (int j = begin; j < end; ++j)
        {
          float Ij = eventTimes[TColInd[j]];
          float Nj = eventTimes[TColInd[j] + eventTimesPitch];

          float jMaxSuscep;
          jMaxSuscep = fminf(Nj, Ij);
          jMaxSuscep = fminf(jMaxSuscep, obsTime);
          float exposure = fminf(Ni, jMaxSuscep) - fminf(Ii, jMaxSuscep);
          exposure += gamma2 * (fminf(Ri, jMaxSuscep) - fminf(Ni, jMaxSuscep));
          TVal[j] = exposure;
        }
    }
}

__global__ void
calcT(const int infecSize, const int nnz, int* TRowPtr, int* TColInd,
    float* TVal, float* eventTimes, const int eventTimesPitch,
    const float gamma2, const float obsTime)
{
  // Each warp calculates a row i of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int wid = tid / 32; // Warp id
  int lane = tid & (32 - 1); // Id within a warp

  int row = wid;

  if (row < infecSize)
    {
      int begin = TRowPtr[row];
      int end = TRowPtr[row + 1];

      float Ii = eventTimes[row]; // First column  -- argument for row-major here, I would have thought.
      float Ni = eventTimes[row + eventTimesPitch]; // Second column
      float Ri = eventTimes[row + eventTimesPitch * 2]; // Third column

      for (int jj = begin + lane; jj < end; jj += 32)
        {
          float Ij = eventTimes[TColInd[jj]];
          float Nj = eventTimes[TColInd[jj] + eventTimesPitch];

          float jMaxSuscep;
          jMaxSuscep = fminf(Nj, Ij);
          jMaxSuscep = fminf(jMaxSuscep, obsTime);
          float exposure = fminf(Ni, jMaxSuscep) - fminf(Ii, jMaxSuscep);
          exposure += gamma2 * (fminf(Ri, jMaxSuscep) - fminf(Ni, jMaxSuscep));
          TVal[jj] = exposure;
        }
    }
}

__global__ void
calcTShared(const int infecSize, const int nnz, int* TRowPtr, int* TColInd,
    float* TVal, float* eventTimes, const int eventTimesPitch,
    const float gamma2, const float obsTime)
{
  // Each warp calculates a row i of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gwid = tid / 32; // Global Warp id
  int lwid = threadIdx.x / 32; // Local warp ID
  int lane = tid & (32 - 1); // Id within a warp

  __shared__
  int begin[THREADSPERBLOCK / 32];
  __shared__
  int end[THREADSPERBLOCK / 32];
  __shared__
  float Ii[THREADSPERBLOCK / 32];
  __shared__
  float Ni[THREADSPERBLOCK / 32];
  __shared__
  float Ri[THREADSPERBLOCK / 32];

  int row = gwid;

  if (row < infecSize)
    {
      if (lane == 0)
        {
          begin[lwid] = TRowPtr[row];
          end[lwid] = TRowPtr[row + 1];
          Ii[lwid] = eventTimes[row]; // First column  -- argument for row-majorhere, I would have thought.
          Ni[lwid] = eventTimes[row + eventTimesPitch]; // Second column
          Ri[lwid] = eventTimes[row + eventTimesPitch * 2]; // Third column
        }
      __syncthreads();

      for (int jj = begin[lwid] + lane; jj < end[lwid]; jj += 32)
        {
          float Ij = eventTimes[TColInd[jj]];
          float Nj = eventTimes[TColInd[jj] + eventTimesPitch];

          float jMaxSuscep;
          jMaxSuscep = fminf(Nj, Ij);
          jMaxSuscep = fminf(jMaxSuscep, obsTime);
          float exposure = fminf(Ni[lwid], jMaxSuscep)
              - fminf(Ii[lwid], jMaxSuscep);
          exposure += gamma2
              * (fminf(Ri[lwid], jMaxSuscep) - fminf(Ni[lwid], jMaxSuscep));
          TVal[jj] = exposure;
        }
    }
}

__global__ void
_calcIntegral(const int infecSize, int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    const float obsTime, float* output)
{
  // Each warp calculates a row i of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gwid = tid / 32; // Global Warp id
  int lwid = threadIdx.x / 32; // Local warp ID
  int lane = tid & (32 - 1); // Id within a warp

  __shared__
  int begin[THREADSPERBLOCK / 32];
  __shared__
  int end[THREADSPERBLOCK / 32];
  __shared__
  float Ii[THREADSPERBLOCK / 32];
  __shared__
  float Ni[THREADSPERBLOCK / 32];
  __shared__
  float Ri[THREADSPERBLOCK / 32];
  __shared__
  float buff[THREADSPERBLOCK];

  buff[threadIdx.x] = 0.0;

  int row = gwid;

  if (row < infecSize)
    {
      if (lane == 0)
        {
          begin[lwid] = DRowPtr[row];
          end[lwid] = DRowPtr[row + 1];
          Ii[lwid] = eventTimes[row]; // First column  -- argument for row-major here, I would have thought.
          Ni[lwid] = eventTimes[row + eventTimesPitch]; // Second column
          Ri[lwid] = eventTimes[row + eventTimesPitch * 2]; // Third column
        }
      __syncthreads();

      float threadSum = 0.0f;
      for (int jj = begin[lwid] + lane; jj < end[lwid]; jj += 32)
        {
          // Integrated infection pressure
          float Ij = eventTimes[DColInd[jj]];
          float Nj = eventTimes[DColInd[jj] + eventTimesPitch];
          float jMaxSuscep;
          jMaxSuscep = fminf(Nj, Ij);
          jMaxSuscep = fminf(jMaxSuscep, obsTime);
          float betaij = fminf(Ni[lwid], jMaxSuscep)
              - fminf(Ii[lwid], jMaxSuscep);
          betaij += gamma2
              * (fminf(Ri[lwid], jMaxSuscep) - fminf(Ni[lwid], jMaxSuscep));

          // Apply distance kernel and suscep
          betaij *= delta / (delta * delta + D[jj]);
          betaij *= susceptibility[DColInd[jj]];
          threadSum += betaij;
        }
      buff[threadIdx.x] = threadSum*infectivity[row];
    }
  __syncthreads();

  // Reduce all warp sums and write to global memory.
  for (unsigned int size = blockDim.x/2; size > 32; size >>= 1)
    {
      if (threadIdx.x < size)
        buff[threadIdx.x] += buff[threadIdx.x + size];
      __syncthreads();
    }
  if (threadIdx.x < 32) {
      volatile float* vbuff = buff;
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 32];
      vbuff[threadIdx.x] += vbuff[threadIdx.x + 16];
      vbuff[threadIdx.x] += vbuff[threadIdx.x +  8];
      vbuff[threadIdx.x] += vbuff[threadIdx.x +  4];
      vbuff[threadIdx.x] += vbuff[threadIdx.x +  2];
      vbuff[threadIdx.x] += vbuff[threadIdx.x +  1];
  }


  if (threadIdx.x == 0)
    {
      output[blockIdx.x] = buff[0];
    }
}

__global__ void
calcE(const int infecSize, const int nnz, const int* ERowPtr,
    const int* EColInd, float* EVal, const float* eventTimes,
    const int eventTimesPitch, const float gamma2)
{
  // Each warp calculates a row of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gwid = tid / 32; // Global Warp id
  int lane = tid & (32 - 1); // Id within a warp

  int row = gwid;

  if (row < infecSize)
    {
      int begin = ERowPtr[row];
      int end = ERowPtr[row + 1];

      float Ij = eventTimes[row];

      for (int ii = begin + lane; ii < end; ii += 32)
        {
          int i = EColInd[ii];
          float Ii = eventTimes[i];
          float Ni = eventTimes[eventTimesPitch + i];
          float Ri = eventTimes[eventTimesPitch * 2 + i];

          if (Ii < Ij and Ij <= Ni)
            EVal[ii] = 1.0f;
          else if (Ni < Ij and Ij <= Ri)
            EVal[ii] = gamma2;
          else
            EVal[ii] = 0.0f;
        }
    }
}

__global__ void
calcSpecPow(const int size, const int nSpecies, float* specpow,
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
sequence(float* buff, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    buff[idx] = (float) idx;
}

GpuLikelihood::GpuLikelihood(const size_t realPopSize, const size_t popSize,
    const size_t numInfecs, const size_t nSpecies, const float obsTime,
    const size_t distanceNNZ) :
    realPopSize_(realPopSize), popSize_(popSize), numInfecs_(numInfecs), numSpecies_(
        nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(0), sumI_(0), bgIntegral_(
        0.0), covariateCopies_(0), devAnimals_(NULL), animalsPitch_(0), devAnimalsInfPow_(
        NULL), devAnimalsSuscPow_(NULL), devEventTimes_(NULL), devSusceptibility_(
        NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(NULL), devDColInd_(
        NULL), dnnz_(distanceNNZ), devTVal_(NULL), devDTVal_(NULL), devEVal_(
        NULL), devERowPtr_(NULL), devEColInd_(NULL), devTmp_(NULL), epsilon_(
        0.0f), gamma1_(0.0f), gamma2_(0.0f), devXi_(NULL), devPsi_(NULL), devZeta_(
        NULL), devPhi_(NULL), delta_(0.0f)
{

  // Allocate Animals_
  checkCudaError(
      cudaMallocPitch(&devAnimals_, &animalsPitch_, popSize_ * sizeof(float), numSpecies_));
  animalsPitch_ /= sizeof(float);
  checkCudaError(
      cudaMallocPitch(&devAnimalsSuscPow_, &animalsSuscPowPitch_, popSize_ * sizeof(float), numSpecies_));
  animalsSuscPowPitch_ /= sizeof(float);
  checkCudaError(
      cudaMallocPitch(&devAnimalsInfPow_, &animalsInfPowPitch_, numInfecs_ * sizeof(float), numSpecies_));
  animalsInfPowPitch_ /= sizeof(float);

  // Allocate Distance_ CRS matrix
  checkCudaError(cudaMalloc(&devDVal_, dnnz_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devDRowPtr_, (popSize_ + 1) * sizeof(int)));
  checkCudaError(cudaMalloc(&devDColInd_, dnnz_ * sizeof(float)));

  // Set up reference counter to covariate data
  covariateCopies_ = new size_t;
  *covariateCopies_ = 1;

  // Allocate Event times - popSize_ * NUMEVENTS matrix
  checkCudaError(
      cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS));
  eventTimesPitch_ /= sizeof(float);

  // Allocate intermediate T and DT
  checkCudaError(cudaMalloc(&devTVal_, dnnz_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devDTVal_, dnnz_ * sizeof(float)));

  // Allocate intermediate infectivity and susceptibility
  checkCudaError(cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devInfectivity_, numInfecs_ * sizeof(float)));

  // Allocate product vector
  checkCudaError(cudaMalloc(&devProduct_, numInfecs_ * sizeof(float)));

  // Allocate temporary vector
  checkCudaError(cudaMalloc(&devTmp_, popSize_ * sizeof(float)));

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
    realPopSize_(other.realPopSize_), popSize_(other.popSize_), numInfecs_(
        other.numInfecs_), numSpecies_(other.numSpecies_), obsTime_(
        other.obsTime_), I1Time_(other.I1Time_), I1Idx_(other.I1Idx_), sumI_(
        other.sumI_), bgIntegral_(other.bgIntegral_), lp_(other.lp_), covariateCopies_(
        other.covariateCopies_), devAnimals_(other.devAnimals_), animalsPitch_(
        other.animalsPitch_), devDVal_(other.devDVal_), devDRowPtr_(
        other.devDRowPtr_), devDColInd_(other.devDColInd_), dnnz_(other.dnnz_), devERowPtr_(
        other.devERowPtr_), devEColInd_(other.devEColInd_), ennz_(other.ennz_), epsilon_(
        other.epsilon_), gamma1_(other.gamma1_), gamma2_(other.gamma2_), delta_(
        other.delta_)
{
  timeval start, end;
  gettimeofday(&start, NULL);
  // Allocate Animals_
  checkCudaError(
      cudaMallocPitch(&devAnimalsInfPow_, &animalsInfPowPitch_, numInfecs_ * sizeof(float), numSpecies_));
  animalsInfPowPitch_ /= sizeof(float);
  checkCudaError(
      cudaMemcpy2D(devAnimalsInfPow_,animalsInfPowPitch_*sizeof(float),other.devAnimalsInfPow_,other.animalsInfPowPitch_*sizeof(float),numInfecs_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));

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
      cudaMemcpy2D(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),numInfecs_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

  // Allocate and copy intermediate T and DT
  checkCudaError(cudaMalloc(&devTVal_, dnnz_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devTVal_,other.devTVal_,dnnz_*sizeof(float),cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devDTVal_, dnnz_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devDTVal_, other.devDTVal_, dnnz_*sizeof(float), cudaMemcpyDeviceToDevice));

  // Allocate and copy intermediate infectivity and susceptibility
  checkCudaError(cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devSusceptibility_, other.devSusceptibility_, popSize_ * sizeof(float),cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devInfectivity_, numInfecs_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devInfectivity_, other.devInfectivity_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

  // Allocate and copy product vector
  checkCudaError(cudaMalloc(&devProduct_, numInfecs_ * sizeof(float)));
  checkCudaError(
      cudaMemcpy(devProduct_, other.devProduct_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devEVal_, ennz_*sizeof(float)));
  checkCudaError(
      cudaMemcpy(devEVal_, other.devEVal_, ennz_*sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMalloc(&devEDVal_, ennz_*sizeof(float)));
  checkCudaError(
      cudaMemcpy(devEDVal_, other. devEDVal_, ennz_*sizeof(float), cudaMemcpyDeviceToDevice));

  // Allocate temporary vector
  checkCudaError(cudaMalloc(&devTmp_, popSize_ * sizeof(float)));

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
      cudaMemcpy2DAsync(devAnimalsInfPow_,animalsInfPowPitch_*sizeof(float),other.devAnimalsInfPow_,other.animalsInfPowPitch_*sizeof(float),numInfecs_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpy2DAsync(devAnimalsSuscPow_,animalsSuscPowPitch_*sizeof(float),other.devAnimalsSuscPow_,other.animalsSuscPowPitch_*sizeof(float),popSize_*sizeof(float),numSpecies_,cudaMemcpyDeviceToDevice));

  // copy event times - popSize_ * NUMEVENTS matrix
  checkCudaError(
      cudaMemcpy2DAsync(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),numInfecs_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

  // copy intermediate T and DT
  checkCudaError(
      cudaMemcpyAsync(devTVal_,other.devTVal_,dnnz_*sizeof(float),cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devDTVal_, other.devDTVal_, dnnz_*sizeof(float), cudaMemcpyDeviceToDevice));

  // copy intermediate infectivity and susceptibility
  checkCudaError(
      cudaMemcpyAsync(devSusceptibility_, other.devSusceptibility_, popSize_ * sizeof(float),cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devInfectivity_, other.devInfectivity_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

  // copy product vector
  checkCudaError(
      cudaMemcpyAsync(devProduct_, other.devProduct_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devEVal_, other.devEVal_, ennz_*sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devEDVal_, other. devEDVal_, ennz_*sizeof(float), cudaMemcpyDeviceToDevice));

  // Device Parameters Copy
  checkCudaError(
      cudaMemcpyAsync(devXi_, other.devXi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devPsi_, other.devPsi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devZeta_, other.devZeta_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devPhi_, other.devPhi_, numSpecies_ * sizeof(float), cudaMemcpyDeviceToDevice));

  // Host Parameters Copy
  epsilon_ = other.epsilon_;
  gamma1_ = other.gamma1_;
  gamma2_ = other.gamma2_;
  delta_ = other.delta_;

  // Likelihood components
  integral_ = other.integral_;
  bgIntegral_ = other.bgIntegral_;
  lp_ = other.lp_;

  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  cudaDeviceSynchronize();
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
      cudaFree(devERowPtr_);
      cudaFree(devEColInd_);

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
  if (devTVal_)
    cudaFree(devTVal_);
  if (devDTVal_)
    cudaFree(devDTVal_);
  if (devSusceptibility_)
    cudaFree(devSusceptibility_);
  if (devInfectivity_)
    cudaFree(devInfectivity_);
  if (devTmp_)
    cudaFree(devTmp_);
  if (devProduct_)
    cudaFree(devProduct_);
  if (devEVal_)
    {
      cudaFree(devEVal_);
      cudaFree(devEDVal_);
    }

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
GpuLikelihood::SetEvents(const float* data)
{
  // Get event times into GPU memory
  cudaError_t rv = cudaMemcpy2D(devEventTimes_,
      eventTimesPitch_ * sizeof(float), data, popSize_ * sizeof(float),
      popSize_ * sizeof(float), NUMEVENTS, cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copying event times to device failed", rv);

  thrust::device_ptr<float> v(devEventTimes_); // REQUIRES COL MAJOR!!
  thrust::device_ptr<float> myMin = thrust::min_element(v, v + numInfecs_);
  I1Idx_ = myMin - v;
  I1Time_ = *myMin;
  sumI_ = thrust::reduce(v, v + numInfecs_, 0.0f, thrust::plus<float>());
}

void
GpuLikelihood::SetSpecies(const float* data)
{
  // Loads species data assuming **COL MAJOR**
  cudaError_t rv = cudaMemcpy2D(devAnimals_, animalsPitch_ * sizeof(float),
      data, popSize_ * sizeof(float), popSize_ * sizeof(float), numSpecies_,
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Failed copying species data to device", rv);

  CalcInfectivity();
  CalcSusceptibility();
}

void
GpuLikelihood::SetDistance(const float* data, const int* rowptr,
    const int* colind)
{
  // Loads distance data into memory
  int rv = cudaMemcpy(devDVal_, data, dnnz_ * sizeof(float),
      cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDRowPtr_, rowptr, (popSize_ + 1) * sizeof(int),
      cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDColInd_, colind, dnnz_ * sizeof(int),
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copy of distance matrix to device failed",
        (cudaError_t) rv);

  std::cerr << "Creating E matrix" << std::endl;
  // Create E_ matrix -- this is an nxn sparse range of the nxm distance matrix
  std::vector<int> eRowPtr;
  std::vector<int> tmpColInd;
  std::vector<float> tmpVals;
  int eNNZ = 0;
  for (size_t i = 0; i < numInfecs_; ++i)
    {
      eRowPtr.push_back(eNNZ);
      int begin = rowptr[i];
      int end = rowptr[i + 1];
      for (size_t jj = begin; jj < end; ++jj)
        {
          if (colind[jj] < numInfecs_)
            {
              tmpColInd.push_back(colind[jj]);
              tmpVals.push_back(data[jj]);
              eNNZ++;
            }
        }
    }
  eRowPtr.push_back(eNNZ);
  std::cerr << "Done\nCopying to device memory" << std::endl;

  // Transfer to device
  checkCudaError(cudaMalloc(&devERowPtr_,eRowPtr.size() * sizeof(int)));
  checkCudaError(cudaMalloc(&devEColInd_,tmpColInd.size() * sizeof(int)));
  checkCudaError(cudaMalloc(&devEVal_,tmpVals.size() * sizeof(float)));
  checkCudaError(cudaMalloc(&devEDVal_,tmpVals.size() * sizeof(float)));
  ennz_ = tmpVals.size();

  checkCudaError(
      cudaMemcpy(devERowPtr_,eRowPtr.data(),eRowPtr.size() * sizeof(int),cudaMemcpyHostToDevice));
  checkCudaError(
      cudaMemcpy(devEColInd_,tmpColInd.data(),tmpColInd.size() * sizeof(int),cudaMemcpyHostToDevice));
}

void
GpuLikelihood::SetParameters(float* epsilon, float* gamma1, float* gamma2,
    float* xi, float* psi, float* zeta, float* phi, float* delta)
{
  epsilon_ = *epsilon;
  gamma1_ = *gamma1;
  gamma2_ = *gamma2;
  delta_ = *delta;

  cudaError_t rv = cudaMemcpy(devXi_, xi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Error copying xi to GPU", rv);

  rv = cudaMemcpy(devPsi_, psi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Error copying psi to GPU", rv);

  rv = cudaMemcpy(devZeta_, zeta, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Error copying zeta to GPU", rv);

  rv = cudaMemcpy(devPhi_, phi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Error copying psi to GPU", rv);
}

void
GpuLikelihood::CalcEvents()
{
  // Calculates the T_ matrix -- sparse matrix operation!
  cudaGetLastError();
  size_t blocksPerGrid = (dnnz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  std::cerr << "Calculating Events with block size "
      << THREADSPERBLOCK << " and blocks per grid " << blocksPerGrid
      << std::endl;
calcTShared<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, dnnz_, devDRowPtr_, devDColInd_, devTVal_, devEventTimes_, eventTimesPitch_, gamma2_, obsTime_);
    checkCudaError(cudaGetLastError());

  // Calculate the E_ matrix
  blocksPerGrid = (ennz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
calcE<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, ennz_, devERowPtr_, devEColInd_, devEVal_, devEventTimes_, eventTimesPitch_, gamma2_);
    checkCudaError(cudaGetLastError());

}

inline
void
GpuLikelihood::CalcInfectivityPow()
{
  int dimBlock(THREADSPERBLOCK);
  int dimGrid((numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
calcSpecPow<<<dimGrid, dimBlock>>>(numInfecs_,numSpecies_,devAnimalsInfPow_, animalsInfPowPitch_,devAnimals_,animalsPitch_,devPsi_);
          checkCudaError(cudaGetLastError());
}

inline
void
GpuLikelihood::CalcInfectivity()
{

  // Now calculate infectivity
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, numInfecs_, numSpecies_,
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
GpuLikelihood::CalcDistance()
{

  // Apply distance kernel to D_, place result in DT_.
  cudaGetLastError();
  size_t blocksPerGrid = (dnnz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcDT<<<blocksPerGrid,THREADSPERBLOCK>>>(devDVal_,devTVal_,dnnz_,devDTVal_, delta_);

  //cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  checkCudaError(err);

  // Apply distance kernel to E_, place result in a temporary
  blocksPerGrid = (ennz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
calcED<<<blocksPerGrid, THREADSPERBLOCK>>>(devEVal_, devERowPtr_, devEColInd_, devDVal_, devDRowPtr_, devDColInd_, devEDVal_, numInfecs_, delta_);
    checkCudaError(cudaGetLastError());
}

void
GpuLikelihood::CalcBgIntegral()
{
  // Get I1Time

  bgIntegral_ = sumI_ - I1Time_ * numInfecs_;
  bgIntegral_ += (obsTime_ - I1Time_) * (realPopSize_ - numInfecs_);
  bgIntegral_ *= epsilon_;

  //std::cerr << "Cuda sumI = " << sumI_ << std::endl;
  //std::cerr << "Cuda I1 = " << I1Time_ << std::endl;
}

void
GpuLikelihood::CalcProduct()
{
  // Calculate Product
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      numInfecs_, numInfecs_, UNITY, crsDescr_, devEDVal_, devERowPtr_,
      devEColInd_, devInfectivity_, ZERO, devTmp_);
  if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      std::cerr << "ED*S failed: " << sparseStat_ << "\n";
    }

  int blocksPerGrid = (numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
calcProdSusceptibility<<<blocksPerGrid, THREADSPERBLOCK>>>(devTmp_, devSusceptibility_, devProduct_, numInfecs_, I1Idx_, epsilon_, gamma1_);
          checkCudaError(cudaGetLastError());

  thrust::device_ptr<float> rb(devProduct_);
  lp_ = thrust::transform_reduce(rb, rb + numInfecs_, Log<float>(), 0.0f,
      thrust::plus<float>());
}

void
GpuLikelihood::CalcIntegral()
{
  integral_ = 0.0f;
  // DT * Susceptibility
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      numInfecs_, popSize_, UNITY, crsDescr_, devDTVal_, devDRowPtr_,
      devDColInd_, devSusceptibility_, ZERO, devTmp_);
  if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      std::cerr << "Error in cusparseScsrmv() " << sparseStat_ << std::endl;
    }

  cudaDeviceSynchronize();
  // infectivity * devTmp
  blasStat_ = cublasSdot(cudaBLAS_, numInfecs_, devInfectivity_, 1, devTmp_, 1,
      &integral_); // May have an issue with 1-based indexing here!
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in cublasSdot() " << blasStat_ << std::endl;
    }
  cudaDeviceSynchronize();

  integral_ *= gamma1_;
}

void
GpuLikelihood::FullCalculate()
{

  timeval start, end;
  gettimeofday(&start, NULL);
  CalcEvents();
  CalcInfectivityPow();
  CalcInfectivity();
  CalcSusceptibilityPow();
  CalcSusceptibility();
  CalcDistance();

  CalcProduct();
  CalcIntegral();
  CalcBgIntegral();

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_ << std::endl;
}

void
GpuLikelihood::Calculate()
{
  timeval start, end;
  CalcEvents();
  gettimeofday(&start, NULL);
  CalcInfectivity();
  CalcSusceptibility();
  CalcDistance();

  CalcProduct();
  CalcIntegral();
  CalcBgIntegral();

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_ << std::endl;
}

void
GpuLikelihood::NewCalculate()
{
  timeval start, end;
  int numRequiredThreads = numInfecs_*32; // One warp per infection
  int blocksPerGrid = (numRequiredThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  thrust::device_vector<float> output(blocksPerGrid);
  std::cerr << "Launching _calcIntegral with " << numRequiredThreads << " threads in " << blocksPerGrid << " blocks per grid" << std::endl;
  gettimeofday(&start, NULL);
  CalcInfectivity();
  CalcSusceptibility();

  _calcIntegral<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_,devDRowPtr_,devDColInd_,devDVal_,
      devEventTimes_,eventTimesPitch_,devSusceptibility_,devInfectivity_,gamma2_,delta_,obsTime_,output.data().get());
  checkCudaError(cudaGetLastError());

  double integral = thrust::reduce(output.begin(), output.end()) * gamma1_;

  CalcProduct();
  CalcBgIntegral();

  logLikelihood_ = lp_ - (integral + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_ << std::endl;
}

void
GpuLikelihood::UpdateInfectionTime(const int idx, const float newTime)
{
  //updateInfectionTime<<<  >>>()
}

float
GpuLikelihood::LogLikelihood() const
{

  return logLikelihood_;
}

