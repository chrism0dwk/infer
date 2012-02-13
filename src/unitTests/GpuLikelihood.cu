/*
 * GpuLikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <iostream>
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
    msg_ += ")";
  }
  ~GpuRuntimeError() throw() {};
  const char*
  what() const throw()
  {
    return msg_.c_str();
  }

private:
  std::string msg_;
};

// CUDA kernels

__device__ __constant__ GpuParams devParams;

__global__ void
calcDT(float* D, float* T, int N, float* DT)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    DT[idx] = devParams.delta / (devParams.delta * devParams.delta + D[idx]) * T[idx];
}

__global__ void
calcT(int nrows, int nnz, int* TRowPtr, int* TColInd, float* TVal, float* eventTimes, int popSize)
{
  // Each thread calculates a row i of the sparse matrix -- probably not efficient!

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int begin = TRowPtr[i];
  int end = TRowPtr[i + 1];

  float Ii = eventTimes[i];
  float Ni = eventTimes[popSize + i];
  float Ri = eventTimes[popSize*2 + i];

  if (i < nrows)
    {
      for(int j=begin; j<end; ++j) {
          float Ij = eventTimes[TColInd[j]];
          float exposure = fminf(Ni, Ij) - fminf(Ii, Ij);
          exposure += devParams.gamma2 * (fminf(Ri, Ij) - fminf(Ni, Ij));
          *(TVal + j) = exposure;
      }
    }
}


__global__ void
calcSpecPow(int size, int nSpecies, float* specpow, const float* animals, const float* powers)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < size and col < nSpecies) {
      specpow[col*size + row] = powf(animals[col*size + row],powers[col]);
  }
}


__global__ void
setto(float* buff, int size, float val)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < size)
    buff[row + size*col] = val;
}



GpuLikelihood::GpuLikelihood(const size_t popSize, const size_t numInfecs, const size_t nSpecies, const size_t distanceNNZ) :
    popSize_(popSize), numInfecs_(numInfecs), numSpecies_(nSpecies), devAnimals_(NULL), devAnimalsInfPow_(
        NULL), devAnimalsSuscPow_(NULL), devEventTimes_(NULL), devSusceptibility_(
        NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(NULL), devDColInd_(
        NULL), dnnz_(distanceNNZ), devTVal_(NULL), devDTVal_(NULL), devEVal_(NULL), devEColPtr_(
        NULL), devERowInd_(NULL), devTmp_(NULL), epsilon_(NULL), gamma1_(NULL), gamma2_(
        NULL), xi_(NULL), psi_(NULL), zeta_(NULL), phi_(NULL), delta_(NULL)
{

  int rv;

  // Allocate Event times - popSize_ * NUMEVENTS matrix
  rv = cudaMalloc(&devEventTimes_, numInfecs_ * NUMEVENTS * sizeof(float));
  if(rv != cudaSuccess) throw GpuRuntimeError("Cannot allocate memory for event times", (cudaError_t)rv);

  // Allocate Animals_
  rv = cudaMalloc(&devAnimals_, popSize_ * numSpecies_ * sizeof(float));
  rv |= cudaMalloc(&devAnimalsSuscPow_, popSize_ * numSpecies_ * sizeof(float));
  rv |= cudaMalloc(&devAnimalsInfPow_, numInfecs_ * numSpecies_ * sizeof(float));
  if(rv != cudaSuccess) throw GpuRuntimeError("Cannot allocate memory for animals", (cudaError_t)rv);

  // Allocate Distance_ CRS matrix
  rv = cudaMalloc(&devDVal_, dnnz_ * sizeof(float));
  rv |= cudaMalloc(&devDRowPtr_, (popSize_+1) * sizeof(int));
  rv |= cudaMalloc(&devDColInd_, dnnz_ * sizeof(float));
  if(rv != cudaSuccess) throw GpuRuntimeError("Cannot allocate distance matrix", (cudaError_t)rv);

  // Allocate intermediate T and DT
  rv = cudaMalloc(&devTVal_, dnnz_ * sizeof(float));
  rv |= cudaMalloc(&devDTVal_, dnnz_ * sizeof(float));

  // Allocate intermediate infectivity and susceptibility
  rv = cudaMalloc(&devSusceptibility_,popSize_ * sizeof(float));
  rv |= cudaMalloc(&devInfectivity_,numInfecs_ * sizeof(float));

  // Allocate temporary vector
  rv = cudaMalloc(&devTmp_,numInfecs_ * sizeof(float));
  if(rv != cudaSuccess) throw GpuRuntimeError("Cannot allocate temporary structures", (cudaError_t)rv);

  // BLAS handles
  blasStat_ = cublasCreate(&cudaBLAS_);
  if(blasStat_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUBLAS init failed");

  sparseStat_ = cusparseCreate(&cudaSparse_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("CUSPARSE init failed");

  sparseStat_ = cusparseCreateMatDescr(&crsDescr_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error("CUSPARSE matrix descriptor init failed");
  cusparseSetMatType(crsDescr_,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(crsDescr_,CUSPARSE_INDEX_BASE_ZERO);

}

GpuLikelihood::~GpuLikelihood()
{
  if(devEventTimes_ != NULL) cudaFree(devEventTimes_);
  if(devAnimals_ != NULL) cudaFree(devAnimals_);
  if(devAnimalsSuscPow_ != NULL) cudaFree(devAnimalsSuscPow_);
  if(devAnimalsInfPow_ != NULL) cudaFree(devAnimalsInfPow_);
  if(devDVal_ != NULL) cudaFree(devDVal_);
  if(devDRowPtr_ != NULL) cudaFree(devDRowPtr_);
  if(devDColInd_ != NULL) cudaFree(devDColInd_);
  if(devTVal_ != NULL) cudaFree(devTVal_);
  if(devDTVal_ != NULL) cudaFree(devDTVal_);
  if(devSusceptibility_ != NULL) cudaFree(devSusceptibility_);
  if(devInfectivity_ != NULL) cudaFree(devInfectivity_);
  if(devTmp_ != NULL) cudaFree(devTmp_);

  cublasDestroy(cudaBLAS_);
  cusparseDestroy(cudaSparse_);
}

void
GpuLikelihood::SetEvents(const float* data)
{
  // Get event times into GPU memory
  cudaError_t rv = cudaMemcpy(devEventTimes_,data,numInfecs_ * NUMEVENTS * sizeof(float), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) throw GpuRuntimeError("Copying event times to device failed", rv);
}

void
GpuLikelihood::SetSpecies(const float* data)
{
  // Loads species data assuming **COL MAJOR**
  cudaError_t rv = cudaMemcpy(devAnimals_,data,popSize_ * numSpecies_ * sizeof(float),cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) throw GpuRuntimeError("Copy species matrix to device failed", rv);

}

void
GpuLikelihood::SetDistance(const float* data, const int* rowptr, const int* colind)
{
  // Loads distance data into memory
  int rv = cudaMemcpy(devDVal_,data,dnnz_ * sizeof(float), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDRowPtr_,rowptr,(popSize_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDColInd_,colind,dnnz_ * sizeof(int), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) throw GpuRuntimeError("Copy of distance matrix to device failed",(cudaError_t)rv);
}

void
GpuLikelihood::SetParameters(GpuParams params)
{
  gpuParams_ = params;
  cudaError_t rv = cudaMemcpyToSymbol("devParams",&params,sizeof(GpuParams));
  if(rv != cudaSuccess) throw GpuRuntimeError("Error copying param values to device", rv);
}

void
GpuLikelihood::SetParameters(float* epsilon, float* gamma1, float* gamma2, float* xi, float* psi,  float* zeta, float* phi, float* delta)
{
  gpuParams_.epsilon = *epsilon;
  gpuParams_.gamma1 = *gamma1;
  gpuParams_.gamma2 = *gamma2;
  gpuParams_. delta = *delta;

  for(size_t p = 0; p<NUMSPECIES; ++p)
    {
      gpuParams_.xi[p] = xi[p];
      gpuParams_.psi[p] = psi[p];
      gpuParams_.zeta[p] = zeta[p];
      gpuParams_.phi[p] = phi[p];
    }

  SetParameters(gpuParams_);
}

void
GpuLikelihood::CalcEvents()
{
  // Calculates the T_ matrix -- sparse matrix operation!
  size_t blocksPerGrid = (dnnz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcT<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, dnnz_, devDRowPtr_, devDColInd_, devTVal_, devEventTimes_, popSize_);
}

void
GpuLikelihood::CalcInfectivity()
{
  // Calculates infectivity powers and sums over infectivity

  dim3 myBlockDim(1,THREADSPERBLOCK);
  dim3 myGridDim(3,(numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  std::cerr << "Grid: (" << myGridDim.x << "," << myGridDim.y << ")" << std::endl;
  std::cerr << "Block: (" << myBlockDim.x << "," << myBlockDim.y << ")" << std::endl;
  setto<<<myGridDim, myBlockDim>>>(devAnimalsInfPow_,numInfecs_,2.0);

  float* tmp = new float[numInfecs_*3];
  cudaMemcpy(tmp,devAnimalsInfPow_,numInfecs_*numSpecies_*sizeof(float),cudaMemcpyDeviceToHost);
  for(size_t i=0; i<numInfecs_*3; ++i) std::cerr << tmp[i] << ", ";
  std::cerr << std::endl;
  delete[] tmp;

  // First do devAnimalsInfPow_
  dim3 dimBlock(3,THREADSPERBLOCK);
  dim3 dimGrid(1, (numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  calcSpecPow<<<dimGrid, dimBlock>>>(numInfecs_,numSpecies_,devAnimalsInfPow_,devAnimals_,devParams.psi);

  tmp = new float[numInfecs_*3];
  cudaMemcpy(tmp,devAnimalsInfPow_,numInfecs_*numSpecies_*sizeof(float),cudaMemcpyDeviceToHost);
  for(size_t i=0; i<numInfecs_*3; ++i) std::cerr << tmp[i] << ", ";
  std::cerr << std::endl;
  delete[] tmp;

  // Now calculate infectivity
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, numInfecs_, numSpecies_, &UNITY,
        devAnimalsInfPow_, numInfecs_, devParams.xi, 1, &ZERO,
        devInfectivity_, 1);
    if (blasStat_ != CUBLAS_STATUS_SUCCESS)
      {
        std::cerr << "Error in infectivity: " << blasStat_ << std::endl;
      }



}


void
GpuLikelihood::CalcSusceptibility()
{
  // Calculates susceptibility powers and sums over suscept.

  dim3 dimBlock(3,THREADSPERBLOCK);
  dim3 dimGrid(1, (popSize_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  calcSpecPow<<<dimGrid, dimBlock>>>(popSize_,NUMSPECIES,devAnimalsSuscPow_,devAnimals_,devParams.phi);

  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, popSize_, numSpecies_, &UNITY,
        devAnimalsSuscPow_, popSize_, devParams.zeta, 1, &ZERO,
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
  size_t blocksPerGrid = (dnnz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcDT<<<blocksPerGrid,THREADSPERBLOCK>>>(devDVal_,devTVal_,dnnz_,devDTVal_);

}

void
GpuLikelihood::Calculate()
{

  CalcEvents();
  CalcInfectivity();
  CalcSusceptibility();
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
//  blasStat_ = cublasSdot(cudaBLAS_, numInfecs_, devInfectivity_, 1, devTmp_,
//      1, &logLikelihood_); // May have an issue with 1-based indexing here!
//  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
//    {
//      std::cerr << "Error in cublasSdot() " << blasStat_ << std::endl;
//    }
  cudaDeviceSynchronize();
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
//  blasStat_ = cublasSdot(cudaBLAS_, numInfecs_, devInfectivity_, 1, devTmp_,
//      1, &logLikelihood_); // May have an issue with 1-based indexing here!
//  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
//    {
//      std::cerr << "Error in cublasSdot() " << blasStat_ << std::endl;
//    }
  cudaDeviceSynchronize();
}

float
GpuLikelihood::LogLikelihood() const
{
  return logLikelihood_;
}
