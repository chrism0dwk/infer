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


#define checkCudaError(err)  __checkCudaError(err, __FILE__, __LINE__)

void
__checkCudaError(const cudaError_t err, const char* file, const int line)
{
  if(err != cudaSuccess) {
    std::stringstream s;
    s << file << "(" << line
      << ") : Cuda Runtime error ";
    throw GpuRuntimeError(s.str(),err);
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
calcDT(const float* D, const float* T, const int N, float* DT, const float delta)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    DT[idx] = delta / (delta * delta + D[idx]) * T[idx];
}


__global__ void
calcED(const float* eVal, const int* eRowPtr, const int* eColInd, const float* dVal,
    const int* dRowPtr, const int* dColInd, float* edVal, const int numInfecs, const float delta)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int wid = tid / 32; // Warp id
  int lane = tid & (32 - 1); // Id within a warp

  int row = wid;
  if (row < numInfecs)
    {
      int d_begin = dRowPtr[row]; //int d_end = dRowPtr[row+1];
      int e_begin = eRowPtr[row]; int e_end = eRowPtr[row+1];
      int rowLen = e_end - e_begin;

      for(int col = lane; col < rowLen; col += 32) // Loop over row in e
        {
          float d = dVal[d_begin+col];
          float e = eVal[e_begin+col];
          edVal[e_begin+col] = delta / (delta*delta + d) * e;

        }
    }
}


__global__ void
calcProdSusceptibility(const float* input, const float* infectivity, float* product, const int size, const int I1, const float epsilon, const float gamma1)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == I1) product[tid] = 1.0;
  else if(tid < size)
    {
      product[tid] = infectivity[tid] * input[tid] * gamma1 + epsilon;
    }
}

__global__ void
calcT(const int infecSize, const int nnz, int* TRowPtr, int* TColInd,
    float* TVal, float* eventTimes, const int eventTimesPitch, const float gamma2, const float obsTime)
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
calcE(const int infecSize, const int nnz, const int* ERowPtr, const int* EColInd,
    float* EVal, const float* eventTimes, const int eventTimesPitch, const float gamma2)
{
  // Each thread calculates a row i of the sparse matrix -- not efficient!

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < infecSize)
    {
      int begin = ERowPtr[i];
      int end = ERowPtr[i + 1];

      float Ii = eventTimes[i]; // First column  -- argument for row-major here, I would have thought.
      float Ni = eventTimes[eventTimesPitch + i]; // Second column
      float Ri = eventTimes[eventTimesPitch * 2 + i]; // Third column

      for (int j = begin; j < end; ++j)
        {
          float Ij = eventTimes[EColInd[j]];

          if(Ii < Ij and Ij <= Ni) EVal[j] = 1.0f;
          else if(Ni < Ij and Ij <= Ri ) EVal[j] = gamma2;
          else EVal[j] = 0.0f;
        }
    }
}



__global__ void
calcSpecPow(const int size, const int nSpecies, float* specpow, const int specpowPitch,
    const float* animals, const int animalsPitch, const float* powers)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size)
    {
      for (unsigned int col=0; col<nSpecies; ++col)
        {
          specpow[col * specpowPitch + row] = powf(animals[col * animalsPitch + row],
              powers[col]);
        }
    }
}


__global__ void
sequence(float* buff, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size) buff[idx] = (float)idx;
}


GpuLikelihood::GpuLikelihood(const size_t realPopSize, const size_t popSize,
    const size_t numInfecs, const size_t nSpecies, const float obsTime,
    const size_t distanceNNZ) :
    realPopSize_(realPopSize), popSize_(popSize), numInfecs_(numInfecs), numSpecies_(
        nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(0), sumI_(0),bgIntegral_(0.0), devAnimals_(
        NULL), animalsPitch_(0), devAnimalsInfPow_(NULL), devAnimalsSuscPow_(NULL), devEventTimes_(
        NULL), devSusceptibility_(NULL), devInfectivity_(NULL), devDVal_(NULL), devDRowPtr_(
        NULL), devDColInd_(NULL), dnnz_(distanceNNZ), devTVal_(NULL), devDTVal_(
        NULL), devEVal_(NULL), devERowPtr_(NULL), devEColInd_(NULL), devTmp_(
        NULL), epsilon_(0.0f), gamma1_(0.0f), gamma2_(0.0f), devXi_(NULL), devPsi_(
        NULL), devZeta_(NULL), devPhi_(NULL), delta_(0.0f)
{

  int rv;

  std::cerr << "Pop size: " << popSize_ << std::endl;
  std::cerr << "Num infected: " << numInfecs_ << std::endl;
  std::cerr << "Num species: " << numSpecies_ << std::endl;

  rv = cudaDeviceReset();
  if(rv != cudaSuccess) throw GpuRuntimeError("Failed to reset device: ",(cudaError_t)rv);

  // Allocate Event times - popSize_ * NUMEVENTS matrix
  rv = cudaMallocPitch(&devEventTimes_, &eventTimesPitch_, popSize_ * sizeof(float), NUMEVENTS);
  eventTimesPitch_ /= sizeof(float);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for event times",
        (cudaError_t) rv);

  // Allocate Animals_
  rv = cudaMallocPitch(&devAnimals_, &animalsPitch_, popSize_ * sizeof(float), numSpecies_);
  animalsPitch_ /= sizeof(float);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for animals",
        (cudaError_t) rv);
  rv = cudaMallocPitch(&devAnimalsSuscPow_, &animalsSuscPowPitch_, popSize_ * sizeof(float), numSpecies_);
  animalsSuscPowPitch_ /= sizeof(float);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Cannot allocate memory for animals susc pow",
        (cudaError_t) rv);
  rv = cudaMallocPitch(&devAnimalsInfPow_, &animalsInfPowPitch_, numInfecs_ * sizeof(float), numSpecies_);
  animalsInfPowPitch_ /= sizeof(float);
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

  // Allocate product vector
  rv = cudaMalloc(&devProduct_, numInfecs_ * sizeof(float));

  // Allocate temporary vector
  rv = cudaMalloc(&devTmp_, popSize_ * sizeof(float));
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
  if(devProduct_ != NULL)
    cudaFree(devProduct_);
  if(devEVal_ != NULL) {
    cudaFree(devEVal_);
    cudaFree(devERowPtr_);
    cudaFree(devEColInd_);
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

  cublasDestroy(cudaBLAS_);
  cusparseDestroy(cudaSparse_);
}

void
GpuLikelihood::SetEvents(const float* data)
{
  // Get event times into GPU memory
  cudaError_t rv = cudaMemcpy2D(devEventTimes_, eventTimesPitch_*sizeof(float), data, popSize_*sizeof(float), popSize_*sizeof(float), NUMEVENTS, cudaMemcpyHostToDevice);
  if (rv != cudaSuccess)
    throw GpuRuntimeError("Copying event times to device failed", rv);

  thrust::device_ptr<float> v(devEventTimes_); // REQUIRES COL MAJOR!!
  thrust::device_ptr<float> myMin = thrust::min_element(v,v+numInfecs_);
  I1Idx_ = myMin - v;
  I1Time_ = *myMin;
  sumI_ = thrust::reduce(v,v+numInfecs_,0.0f,thrust::plus<float>());
}

void
GpuLikelihood::SetSpecies(const float* data)
{
  // Loads species data assuming **COL MAJOR**
  cudaError_t rv = cudaMemcpy2D(devAnimals_, animalsPitch_*sizeof(float), data, popSize_*sizeof(float), popSize_*sizeof(float), numSpecies_, cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) throw GpuRuntimeError("Failed copying species data to device",rv);

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
  std::vector< int > eRowPtr;
  std::vector< int > tmpColInd;
  std::vector< float > tmpVals;
  int eNNZ = 0;
  for(size_t i=0; i<numInfecs_; ++i)
    {
      eRowPtr.push_back(eNNZ);
      int begin = rowptr[i];
      int end = rowptr[i+1];
      for(size_t jj = begin; jj<end; ++jj)
        {
          if(colind[jj] < numInfecs_)
            {
              tmpColInd.push_back(colind[jj]);
              tmpVals.push_back(data[begin+jj]);
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

  checkCudaError(cudaMemcpy(devERowPtr_,eRowPtr.data(),eRowPtr.size() * sizeof(int),cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(devEColInd_,tmpColInd.data(),tmpColInd.size() * sizeof(int),cudaMemcpyHostToDevice));

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
  calcT<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, dnnz_, devDRowPtr_, devDColInd_, devTVal_, devEventTimes_, eventTimesPitch_, gamma2_, obsTime_);

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw GpuRuntimeError("Error calculating events", err);

  // Calculate the E_ matrix
  blocksPerGrid = (ennz_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcE<<<blocksPerGrid,THREADSPERBLOCK>>>(numInfecs_, ennz_, devERowPtr_, devEColInd_, devEVal_, devEventTimes_, eventTimesPitch_, gamma2_);

  // Calculate I1Time_, I1Idx_, and sumI

  cudaThreadSynchronize();

}


inline
void
GpuLikelihood::CalcInfectivityPow()
{
  int dimBlock(THREADSPERBLOCK);
  int dimGrid((numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK);
  calcSpecPow<<<dimGrid, dimBlock>>>(numInfecs_,numSpecies_,devAnimalsInfPow_, animalsInfPowPitch_,devAnimals_,animalsPitch_,devPsi_);

  //cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      throw GpuRuntimeError("Launch of infectivity power kernel failed", err);
    }
}


inline
void
GpuLikelihood::CalcInfectivity()
{

  // Now calculate infectivity
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, numInfecs_, numSpecies_,
      &UNITY, devAnimalsInfPow_, animalsInfPowPitch_, devXi_, 1, &ZERO, devInfectivity_,
      1);
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

  //cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    {
      throw GpuRuntimeError("Launch of susceptibility power kernel failed",
          err);
    }
}

inline
void
GpuLikelihood::CalcSusceptibility()
{
  // Calculates susceptibility powers and sums over suscept.
  blasStat_ = cublasSgemv(cudaBLAS_, CUBLAS_OP_N, popSize_, numSpecies_, &UNITY,
      devAnimalsSuscPow_, animalsSuscPowPitch_, devZeta_, 1, &ZERO, devSusceptibility_, 1);
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


  // Apply distance kernel to E_, place result in ED_
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
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_TRANSPOSE, numInfecs_, numInfecs_, UNITY, crsDescr_, devEDVal_, devERowPtr_, devEColInd_, devInfectivity_, ZERO, devTmp_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      std::cerr << "ED*S failed: " << sparseStat_ << "\n";
    }

  int blocksPerGrid = (numInfecs_ + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
  calcProdSusceptibility<<<blocksPerGrid, THREADSPERBLOCK>>>(devTmp_, devSusceptibility_, devProduct_, numInfecs_, I1Idx_, epsilon_, gamma1_);
  checkCudaError(cudaGetLastError());

  thrust::device_ptr<float> rb(devProduct_);
  lp_ = thrust::transform_reduce(rb, rb+numInfecs_, Log<float>(), 0.0f, thrust::plus<float>());
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
}

void
GpuLikelihood::Calculate()
{
  CalcInfectivity();
  CalcSusceptibility();
  CalcDistance();

  CalcProduct();
  CalcIntegral();
  CalcBgIntegral();

  logLikelihood_ = lp_ - (integral_ + bgIntegral_);

}

float
GpuLikelihood::LogLikelihood() const
{

  return logLikelihood_;
}

