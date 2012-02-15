/*
 * GpuLikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include <math_functions.h>

#include "GpuLikelihood.hpp"

// Constants
const float UNITY = 1.0;
const float ZERO = 0.0;

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

__global__ void
calcDT(float* D, float* T, int N, float* DT, float delta)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    DT[idx] = delta / (delta * delta + D[idx]) * T[idx];
}

__global__ void
calcT(int infecSize, int nnz, int popSize, int* TRowPtr, int* TColInd,
    float* TVal, float* eventTimes, float gamma2, float obsTime)
{
  // Each thread calculates a row i of the sparse matrix -- probably not efficient!

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < infecSize)
    {
      int begin = TRowPtr[i];
      int end = TRowPtr[i + 1];

      float Ii = eventTimes[i]; // First column  -- argument for row-major here, I would have thought.
      float Ni = eventTimes[popSize + i]; // Second column
      float Ri = eventTimes[popSize * 2 + i]; // Third column

      for (int j = begin; j < end; ++j)
        {
          float Ij = eventTimes[TColInd[j]];
          float Nj = eventTimes[TColInd[j] + popSize];

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
calcSpecPow(const int size, const int nSpecies, float* specpow,
    const float* animals, const int stride, const float* powers)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < size) and (col < nSpecies))
    {
      specpow[col * size + row] = powf(animals[col * stride + row],
          powers[col]);
    }
}

template<unsigned int blockSize, typename Op>
  __global__ void
  reduction(const float* buffer, float* rb, int size)
  {
    extern __shared__ float threadBuff[];

    Op op; // Operation functor to use

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    threadBuff[tid] = op.initval(); // Initialize with default init value

    while (i < size)
      {
        threadBuff[tid] = buffer[i];
        if (i + blockSize < size) threadBuff[tid] = op(threadBuff[tid], buffer[i + blockSize]);
        i += gridSize;
      }
    __syncthreads();

    if (blockSize >= 512)
      {
        if (tid < 256)
          {
            threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 256]);
          }
        __syncthreads();
      }
    if (blockSize >= 256)
      {
        if (tid < 128)
          {
            threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 128]);
          }
        __syncthreads();
      }
    if (blockSize >= 128)
      {
        if (tid < 64)
          {
            threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 64]);
          }
        __syncthreads();
      }

    if (tid < 32)
      {
        if (blockSize >= 64)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 32]);
        if (blockSize >= 32)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 16]);
        if (blockSize >= 16)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 8]);
        if (blockSize >= 8)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 4]);
        if (blockSize >= 4)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 2]);
        if (blockSize >= 2)
          threadBuff[tid] = op(threadBuff[tid], threadBuff[tid + 1]);
      }

    if (tid == 0)
      rb[blockIdx.x] = threadBuff[0];
  }


__global__ void
sequence(float* buff, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) buff[idx] = (float)idx;
}

template<class Op>
  __host__ float
  gpuReduction(const float* deviceBuffer, const int size)
  {
    int numBlocks = (size + 4 - 1) / 4;
    float* buff = new float[numBlocks];
    float* devRB;

    cudaError_t rv = cudaMalloc(&devRB, numBlocks * sizeof(float));
    if (rv != cudaSuccess)
      throw GpuRuntimeError("Cannot allocate memory for reduction buffer", rv);

    reduction<4, Op><<<numBlocks, 4, 4*sizeof(float)>>>(deviceBuffer, devRB, size);

    rv = cudaMemcpy(buff, devRB, numBlocks * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (rv != cudaSuccess)
      throw GpuRuntimeError("Could not retrieve reduction buffer from device",
          rv);

    cudaFree(devRB);

    // Final reduction on host
    float result = buff[0];
    Op op;
    for (size_t i = 1; i < numBlocks; ++i)
      result = op(buff[i],result);

    delete[] buff;

    return result;
  }

GpuLikelihood::GpuLikelihood(const size_t realPopSize, const size_t popSize,
    const size_t numInfecs, const size_t nSpecies, const float obsTime,
    const size_t distanceNNZ) :
    realPopSize_(realPopSize), popSize_(popSize), numInfecs_(numInfecs), numSpecies_(
        nSpecies), obsTime_(obsTime), I1Time_(0.0), bgIntegral_(0.0), devAnimals_(
        NULL), devAnimalsInfPow_(NULL), devAnimalsSuscPow_(NULL), devEventTimes_(
        NULL), devSusceptibility_(NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(
        NULL), devDColInd_(NULL), dnnz_(distanceNNZ), devTVal_(NULL), devDTVal_(
        NULL), devEVal_(NULL), devEColPtr_(NULL), devERowInd_(NULL), devTmp_(
        NULL), epsilon_(NULL), gamma1_(NULL), gamma2_(NULL), devXi_(NULL), devPsi_(
        NULL), devZeta_(NULL), devPhi_(NULL), delta_(NULL)
{

  int rv;

  std::cerr << "Pop size: " << popSize_ << std::endl;
  std::cerr << "Num infected: " << numInfecs_ << std::endl;
  std::cerr << "Num species: " << numSpecies_ << std::endl;

  cudaThreadSynchronize();

  // Allocate Event times - popSize_ * NUMEVENTS matrix
  rv = cudaMalloc(&devEventTimes_, popSize_ * NUMEVENTS * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for event times",
        (cudaError_t) rv);

  // Allocate Animals_
  rv = cudaMalloc(&devAnimals_, popSize_ * numSpecies_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for animals",
        (cudaError_t) rv);
  rv = cudaMalloc(&devAnimalsSuscPow_, popSize_ * numSpecies_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for animals susc pow",
        (cudaError_t) rv);
  rv = cudaMalloc(&devAnimalsInfPow_, numInfecs_ * numSpecies_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for animals inf pow",
        (cudaError_t) rv);

  // Allocate Distance_ CRS matrix
  rv = cudaMalloc(&devDVal_, dnnz_ * sizeof(float));
  rv |= cudaMalloc(&devDRowPtr_, (popSize_ + 1) * sizeof(int));
  rv |= cudaMalloc(&devDColInd_, dnnz_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate distance matrix", (cudaError_t) rv);

  // Allocate intermediate T and DT
  rv = cudaMalloc(&devTVal_, dnnz_ * sizeof(float));
  rv |= cudaMalloc(&devDTVal_, dnnz_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate temporary structures",
        (cudaError_t) rv);

  // Allocate intermediate infectivity and susceptibility
  rv = cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float));
  rv |= cudaMalloc(&devInfectivity_, numInfecs_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate temporary structures",
        (cudaError_t) rv);

  // Allocate temporary vector
  rv = cudaMalloc(&devTmp_, numInfecs_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate temporary structures",
        (cudaError_t) rv);

  // Parameters
  rv = cudaMalloc(&devXi_, numSpecies_ * sizeof(float));
  rv |= cudaMalloc(&devPsi_, numSpecies_ * sizeof(float));
  rv |= cudaMalloc(&devZeta_, numSpecies_ * sizeof(float));
  rv |= cudaMalloc(&devPhi_, numSpecies_ * sizeof(float));
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate device parameters",
        (cudaError_t) rv);

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

GpuLikelihood::~GpuLikelihood()
{
  if (devEventTimes_ != NULL)
    cudaFree(devEventTimes_);
  if (devAnimals_ != NULL)
    cudaFree(devAnimals_);
  if (devAnimalsSuscPow_ != NULL)
    cudaFree(devAnimalsSuscPow_);
  if (devAnimalsInfPow_ != NULL)
    cudaFree(devAnimalsInfPow_);
  if (devDVal_ != NULL)
    cudaFree(devDVal_);
  if (devDRowPtr_ != NULL)
    cudaFree(devDRowPtr_);
  if (devDColInd_ != NULL)
    cudaFree(devDColInd_);
  if (devTVal_ != NULL)
    cudaFree(devTVal_);
  if (devDTVal_ != NULL)
    cudaFree(devDTVal_);
  if (devSusceptibility_ != NULL)
    cudaFree(devSusceptibility_);
  if (devInfectivity_ != NULL)
    cudaFree(devInfectivity_);
  if (devTmp_ != NULL)
    cudaFree(devTmp_);

  if (devXi_)
    cudaFree(devXi_);
  if (devPsi_)
    cudaFree(devPsi_);
  if (devZeta_)
    cudaFree(devZeta_);
  if (devPhi_)
    cudaFree(devPhi_);

  cublasDestroy(cudaBLAS_);
  cusparseDestroy(cudaSparse_);
}

void
GpuLikelihood::SetEvents(const float* data)
{
  // Get event times into GPU memory
  cudaError_t rv = cudaMemcpy(devEventTimes_, data,
      popSize_ * NUMEVENTS * sizeof(float), cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copying event times to device failed", rv);
}

void
GpuLikelihood::SetSpecies(const float* data)
{
  // Loads species data assuming **COL MAJOR**
  cudaError_t rv = cudaMemcpy(devAnimals_, data,
      popSize_ * numSpecies_ * sizeof(float), cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copy species matrix to device failed", rv);

  rv = cudaMemcpy(devAnimalsSuscPow_, data,
      popSize_ * numSpecies_ * sizeof(float), cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copy species matrix to device failed", rv);

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
  calcT<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, dnnz_, popSize_, devDRowPtr_, devDColInd_, devTVal_, devEventTimes_, gamma2_, obsTime_);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw GpuRuntimeError("Error calculating events", err);

//  float* vals = new float[dnnz_];
//  int* rowptr = new int[popSize_];
//  int* colind = new int[dnnz_];
//
//  cudaMemcpy(vals, devTVal_, dnnz_ * sizeof(float), cudaMemcpyDeviceToHost);
//  cudaMemcpy(rowptr, devDRowPtr_, popSize_ * sizeof(int),
//      cudaMemcpyDeviceToHost);
//  cudaMemcpy(colind, devDColInd_, dnnz_ * sizeof(int), cudaMemcpyDeviceToHost);
//
//  std::cerr << "T_: ";
//  for (size_t i = 0; i < dnnz_; ++i)
//    std::cerr << vals[i] << " ";
//  std::cerr << std::endl;
//  std::cerr << "TColInd: ";
//  for (size_t i = 0; i < dnnz_; ++i)
//    std::cerr << colind[i] << " ";
//  std::cerr << std::endl;
//  std::cerr << "TRowPtr: ";
//  for (size_t i = 0; i < popSize_; ++i)
//    std::cerr << rowptr[i] << " ";
//  std::cerr << std::endl;
//
//  delete[] vals;

}

void
GpuLikelihood::CalcInfectivity()
{
  // Calculates infectivity powers and sums over infectivity
  cudaGetLastError(); // Reset error status
  // First do devAnimalsInfPow_
  dim3 dimBlock(3, THREADSPERBLOCK);
  dim3 dimGrid(1, (numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  calcSpecPow<<<dimGrid, dimBlock>>>(numInfecs_,numSpecies_,devAnimalsInfPow_,devAnimals_,popSize_,devPsi_);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      throw GpuRuntimeError("Launch of infectivity power kernel failed", err);
    }

  // Now calculate infectivity
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, numInfecs_, numSpecies_,
      &UNITY, devAnimalsInfPow_, numInfecs_, devXi_, 1, &ZERO, devInfectivity_,
      1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in infectivity: " << blasStat_ << std::endl;
    }

  float res;
  cublasSasum(cudaBLAS_, numInfecs_, devInfectivity_, 1, &res);
  std::cerr << "Sum devInfectivity_ (GPU) = " << res << std::endl;
}

void
GpuLikelihood::CalcSusceptibility()
{
  // Calculates susceptibility powers and sums over suscept.
  cudaGetLastError();
  dim3 dimBlock(3, THREADSPERBLOCK);
  dim3 dimGrid(1, (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  calcSpecPow<<<dimGrid, dimBlock>>>(popSize_,numSpecies_,devAnimalsSuscPow_,devAnimals_,popSize_,devPhi_);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      throw GpuRuntimeError("Launch of susceptibility power kernel failed",
          err);
    }

  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, popSize_, numSpecies_, &UNITY,
      devAnimalsSuscPow_, popSize_, devZeta_, 1, &ZERO, devSusceptibility_, 1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in susceptibility: " << blasStat_ << std::endl;
    }

  float res;
  cublasSasum(cudaBLAS_, popSize_, devSusceptibility_, 1, &res);
  std::cerr << "Sum devSusceptibility_ (GPU) = " << res << std::endl;

}

inline
void
GpuLikelihood::CalcDistance()
{

  // Apply distance kernel to D_, place result in DT_.
  cudaGetLastError();
  size_t blocksPerGrid = (dnnz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcDT<<<blocksPerGrid,THREADSPERBLOCK>>>(devDVal_,devTVal_,dnnz_,devDTVal_, delta_);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw GpuRuntimeError("Distance calculation failed", err);

  float res;
  cublasSasum(cudaBLAS_, dnnz_, devTVal_, 1, &res);
  std::cerr << "Sum devTVal_ (GPU) = " << res << std::endl;

}

void
GpuLikelihood::CalcBgIntegral()
{
  // Get I1Time

  float I1 = gpuReduction<Min<float> >(devEventTimes_, numInfecs_);
  float sumI = gpuReduction<Plus<float> >(devEventTimes_, numInfecs_);

  bgIntegral_ = sumI - I1 * numInfecs_;
  bgIntegral_ += (obsTime_ - I1) * (realPopSize_ - numInfecs_);
  bgIntegral_ *= epsilon_;

  std::cerr << "Cuda sumI = " << sumI << std::endl;
  std::cerr << "Cuda I1 = " << I1 << std::endl;
}

void
GpuLikelihood::Calculate()
{

  CalcEvents();
  CalcInfectivity();
  CalcSusceptibility();
  CalcDistance();
  CalcBgIntegral();

  // DT * Susceptibility
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      numInfecs_, popSize_, dnnz_, &UNITY, crsDescr_, devDTVal_, devDRowPtr_,
      devDColInd_, devSusceptibility_, &ZERO, devTmp_);
  if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      std::cerr << "Error in cusparseScsrmv() " << sparseStat_ << std::endl;
    }
  cudaDeviceSynchronize();

  // infectivity * devTmp
  blasStat_ = cublasSdot(cudaBLAS_, numInfecs_, devInfectivity_, 1, devTmp_, 1,
      &logLikelihood_); // May have an issue with 1-based indexing here!
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in cublasSdot() " << blasStat_ << std::endl;
    }
  cudaDeviceSynchronize();

  logLikelihood_ *= gamma1_;
  logLikelihood_ += bgIntegral_;
  logLikelihood_ *= -1;
}

void
GpuLikelihood::UpdateDistance()
{
  CalcDistance();

  // DT * Susceptibility
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      numInfecs_, popSize_, dnnz_, &UNITY, crsDescr_, devDTVal_, devDRowPtr_,
      devDColInd_, devSusceptibility_, &ZERO, devTmp_);
  if (sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      std::cerr << "Error in cusparseScsrmv() " << sparseStat_ << std::endl;
    }

  // infectivity * devTmp
  blasStat_ = cublasSdot(cudaBLAS_, numInfecs_, devInfectivity_, 1, devTmp_, 1,
      &logLikelihood_); // May have an issue with 1-based indexing here!
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "Error in cublasSdot() " << blasStat_ << std::endl;
    }
  cudaDeviceSynchronize();

  logLikelihood_ *= gamma1_;
  logLikelihood_ += bgIntegral_;
  logLikelihood_ *= -1;
}

float
GpuLikelihood::LogLikelihood() const
{

  return logLikelihood_;
}

