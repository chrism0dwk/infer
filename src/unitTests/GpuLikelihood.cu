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
_calcIntegral(const int infecSize, int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    const float obsTime, float* output)
{
  // Each warp calculates a row i of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int row = tid / 32; // Global Warp id
  int lwid = threadIdx.x / 32; // Local warp ID
  int lane = tid & (32 - 1); // Id within a warp

  __shared__
  float buff[THREADSPERBLOCK];

  buff[threadIdx.x] = 0.0;

  if (row < infecSize)
    {

      int begin = DRowPtr[row];
      int end = DRowPtr[row+1];
      float Ii = eventTimes[row];
      float Ni = eventTimes[row + eventTimesPitch];
      float Ri = eventTimes[row + eventTimesPitch*2];

      float threadSum = 0.0f;
      for (int jj = begin + lane; jj < end; jj += 32)
        {
          // Integrated infection pressure
          float Ij = eventTimes[DColInd[jj]];
          float Nj = eventTimes[DColInd[jj] + eventTimesPitch];
          float jMaxSuscep;
          jMaxSuscep = fminf(Nj, Ij);
          jMaxSuscep = fminf(jMaxSuscep, obsTime);
          float betaij = fminf(Ni, jMaxSuscep)
              - fminf(Ii, jMaxSuscep);
          betaij += gamma2
              * (fminf(Ri, jMaxSuscep) - fminf(Ni, jMaxSuscep));

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
_calcProduct(const int infecSize, const int* DRowPtr,
    const int* DColInd, float* D, const float* eventTimes,
    const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float epsilon,
    const float gamma1, const float gamma2, const float delta, float* prodCache)
{
  // Each warp calculates a row of the sparse matrix

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int gwid = tid / 32; // Global Warp id
  int lane = tid & (32 - 1); // Id within a warp

  __shared__ float threadProdCache[THREADSPERBLOCK];
  threadProdCache[threadIdx.x] = 0.0f;
  int row = gwid;

  if (row < infecSize)
    {
      //threadProdCache[threadIdx.x] = 0.0f;
      int begin = DRowPtr[row];
      int end = DRowPtr[row + 1];

      float Ij = eventTimes[row];


      for (int ii = begin + lane; ii < end and DColInd[ii] < infecSize; ii += 32)
        {
          int i = DColInd[ii];
          float Ii = eventTimes[i];
          float Ni = eventTimes[eventTimesPitch + i];
          float Ri = eventTimes[eventTimesPitch * 2 + i];

          if (Ii < Ij and Ij <= Ni)
            threadProdCache[threadIdx.x] += infectivity[i]*delta / (delta*delta + D[ii]);
          else if (Ni < Ij and Ij <= Ri)
            threadProdCache[threadIdx.x] += gamma2*infectivity[i]*delta / (delta*delta + D[ii]);
          else
            threadProdCache[threadIdx.x] += 0.0f;
        }
      __syncthreads();

      // Reduce semi-products into productCache
      volatile float* vThreadProdCache = threadProdCache;
      if(lane < 16)
        {
          vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 16];
          vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x +  8];
          vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x +  4];
          vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x +  2];
          vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x +  1];
        }
      __syncthreads();

      // Write out to global memory -- we're going to get a bank conflict here!
      if(lane == 0) prodCache[row] = threadProdCache[threadIdx.x]*susceptibility[row]*gamma1 + epsilon;
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
_updateInfectionTimeInteg(const int idx, const float newTime, int* DRowPtr, int* DColInd, float* D,
    float* eventTimes, const int eventTimesPitch, const float* susceptibility,
    const float* infectivity, const float gamma2, const float delta,
    const float obsTime)
{

}

GpuLikelihood::GpuLikelihood(const size_t realPopSize, const size_t popSize,
    const size_t numInfecs, const size_t nSpecies, const float obsTime,
    const size_t distanceNNZ) :
    realPopSize_(realPopSize), popSize_(popSize), numInfecs_(numInfecs), numSpecies_(
        nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(0), sumI_(0), bgIntegral_(
        0.0), covariateCopies_(0), devAnimals_(NULL), animalsPitch_(0), devAnimalsInfPow_(
        NULL), devAnimalsSuscPow_(NULL), devEventTimes_(NULL), devSusceptibility_(
        NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(NULL), devDColInd_(
        NULL), dnnz_(distanceNNZ), epsilon_(
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
  hostDRowPtr_ = new int[popSize_ + 1];

  // Set up reference counter to covariate data
  covariateCopies_ = new size_t;
  *covariateCopies_ = 1;

  // Allocate Event times - popSize_ * NUMEVENTS matrix
  checkCudaError(
      cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS));
  eventTimesPitch_ /= sizeof(float);

  // Allocate intermediate infectivity and susceptibility
  checkCudaError(cudaMalloc(&devSusceptibility_, popSize_ * sizeof(float)));
  checkCudaError(cudaMalloc(&devInfectivity_, numInfecs_ * sizeof(float)));

  // Allocate product cache
  checkCudaError(cudaMalloc(&devProduct_, numInfecs_ * sizeof(float)));

  // Allocate integral array
  int numRequiredThreads = numInfecs_*32; // One warp per infection
  integralBuffSize_ = (numRequiredThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  checkCudaError(cudaMalloc(&devIntegral_, integralBuffSize_*sizeof(float)));

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
        other.devDRowPtr_), devDColInd_(other.devDColInd_), dnnz_(other.dnnz_),
        integralBuffSize_(other.integralBuffSize_), epsilon_(
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

  // Allocate integral array
  checkCudaError(cudaMalloc(&devIntegral_, integralBuffSize_*sizeof(float)));


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

  // copy event times
  checkCudaError(
      cudaMemcpy2DAsync(devEventTimes_,eventTimesPitch_*sizeof(float),other.devEventTimes_,other.eventTimesPitch_*sizeof(float),numInfecs_*sizeof(float), NUMEVENTS, cudaMemcpyDeviceToDevice));

  // copy intermediate infectivity and susceptibility
  checkCudaError(
      cudaMemcpyAsync(devSusceptibility_, other.devSusceptibility_, popSize_ * sizeof(float),cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpyAsync(devInfectivity_, other.devInfectivity_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

  // copy product vector
  checkCudaError(
      cudaMemcpyAsync(devProduct_, other.devProduct_, numInfecs_ * sizeof(float), cudaMemcpyDeviceToDevice));

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
  if (devProduct_)
    cudaFree(devProduct_);
  if (devIntegral_)
    cudaFree(devIntegral_);

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
  checkCudaError(cudaMemcpy(devDVal_, data, dnnz_ * sizeof(float),
      cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(devDRowPtr_, rowptr, (popSize_ + 1) * sizeof(int),
      cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(devDColInd_, colind, dnnz_ * sizeof(int),
      cudaMemcpyHostToDevice));
}

void
GpuLikelihood::SetParameters(float* epsilon, float* gamma1, float* gamma2,
    float* xi, float* psi, float* zeta, float* phi, float* delta)
{
  epsilon_ = *epsilon;
  gamma1_ = *gamma1;
  gamma2_ = *gamma2;
  delta_ = *delta;

  checkCudaError(cudaMemcpy(devXi_, xi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice));

  checkCudaError(cudaMemcpy(devPsi_, psi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice));

  checkCudaError(cudaMemcpy(devZeta_, zeta, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice));

  checkCudaError(cudaMemcpy(devPhi_, phi, numSpecies_ * sizeof(float),
      cudaMemcpyHostToDevice));
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


void
GpuLikelihood::CalcBgIntegral()
{
  // Get I1Time

  thrust::device_ptr<float> v(devEventTimes_); // REQUIRES COL MAJOR!!
  thrust::device_ptr<float> myMin = thrust::min_element(v, v + numInfecs_);
  I1Idx_ = myMin - v;
  I1Time_ = *myMin;
  sumI_ = thrust::reduce(v, v + numInfecs_, 0.0f, thrust::plus<float>());

  bgIntegral_ = sumI_ - I1Time_ * numInfecs_;
  bgIntegral_ += (obsTime_ - I1Time_) * (realPopSize_ - numInfecs_);
  bgIntegral_ *= epsilon_;
}

inline
void
GpuLikelihood::CalcProduct()
{

  _calcProduct<<<integralBuffSize_,THREADSPERBLOCK>>>(numInfecs_,devDRowPtr_,devDColInd_,devDVal_,
      devEventTimes_,eventTimesPitch_,devSusceptibility_,devInfectivity_,epsilon_,gamma1_,gamma2_,delta_,devProduct_);

  thrust::device_ptr<float> prodPtr(devProduct_);
  prodPtr[I1Idx_] = 1.0;
  lp_ = thrust::transform_reduce(prodPtr, prodPtr+numInfecs_, Log<float>(), 0.0f,thrust::plus<float>());
}


inline
void
GpuLikelihood::CalcIntegral()
{
  _calcIntegral<<<integralBuffSize_,THREADSPERBLOCK>>>(numInfecs_,devDRowPtr_,devDColInd_,devDVal_,
      devEventTimes_,eventTimesPitch_,devSusceptibility_,devInfectivity_,gamma2_,delta_,obsTime_,devIntegral_);
  checkCudaError(cudaGetLastError());

  thrust::device_ptr<float> integPtr(devIntegral_);
  integral_ = thrust::reduce(integPtr, integPtr + integralBuffSize_) * gamma1_;
}

void
GpuLikelihood::FullCalculate()
{

  timeval start, end;
  gettimeofday(&start, NULL);
  CalcInfectivityPow();
  CalcInfectivity();
  CalcSusceptibilityPow();
  CalcSusceptibility();

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
  gettimeofday(&start, NULL);
  CalcInfectivity();
  CalcSusceptibility();
  CalcIntegral();
  CalcProduct();
  CalcBgIntegral();
  logLikelihood_ = lp_ - (integral_ + bgIntegral_);
  gettimeofday(&end, NULL);
  std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
      << timeinseconds(start, end) << std::endl;
  std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_ << std::endl;
}

void
GpuLikelihood::UpdateInfectionTime(const int idx, const float newTime)
{
  // Require to know number of cols per row -- probably store in host mem.
  int blocksPerGrid = (hostDRowPtr_[idx+1] - hostDRowPtr_[idx] + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  //_updateInfectionTime<<<blocksPerGrid, THREADSPERBLOCK>>>()
}

float
GpuLikelihood::LogLikelihood() const
{

  return logLikelihood_;
}

