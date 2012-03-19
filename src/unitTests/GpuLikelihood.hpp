/*************************************************************************
 *  ./src/unitTests/GpuLikelihood.hpp
 *  Copyright Chris Jewell <chrism0dwk@gmail.com> 2012
 *
 *  This file is part of InFER.
 *
 *  InFER is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  InFER is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with InFER.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************/
/*
 * GpuLikelihood.hpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */

#ifndef GPULIKELIHOOD_HPP_
#define GPULIKELIHOOD_HPP_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


// CUDA defines
#define THREADSPERBLOCK 256

// Model defines
#define NUMEVENTS 3
#define NUMSPECIES 3

class GpuLikelihood
{
public:
  GpuLikelihood(const size_t realPopSize, const size_t popSize, const size_t numInfecs, const size_t maxInfecs, const size_t nSpecies, const float obsTime, const size_t distanceNNZ);
  GpuLikelihood(const GpuLikelihood& other);
  virtual
  ~GpuLikelihood();
  const
  GpuLikelihood&
  operator=(const GpuLikelihood& other);
  void
  SetEvents(const float* data);
  void
  SetSpecies(const float* data);
  void
  SetDistance(const float* data, const int* rowptr, const int* colind);
  void
  SetParameters(float* epsilon, float* gamma1, float* gamma2, float* xi, float* psi, float* zeta, float* phi, float* delta);
  void
  CalcSusceptibilityPow();
  void
  CalcSusceptibility();
  void
  CalcInfectivityPow();
  void
  CalcInfectivity();
  void
  UpdateI1();
  void
  CalcBgIntegral();
  void
  UpdateInfectionTime(const unsigned int idx, const float inTime);
  void
  AddInfectionTime(const unsigned int idx, const float inTime);
  void
  DeleteInfectionTime(const unsigned int idx);
  void
  CalcProduct();
  void
  CalcIntegral();
  void
  FullCalculate();
  void
  Calculate();
  void
  NewCalculate();
  float
  LogLikelihood() const;
  float
  GetN(const int idx) const;
  void
  LazyAddInfecTime(const int idx, const float inTime);

private:
  // Host vars
  const size_t realPopSize_;
  const size_t popSize_;
  size_t numInfecs_;
  size_t maxInfecs_;
  thrust::host_vector<unsigned int> hostInfecIdx_;
  thrust::device_vector<unsigned int> devInfecIdx_;
  const size_t numSpecies_;
  float logLikelihood_;
  const float obsTime_;
  float I1Time_; unsigned int I1Idx_; float sumI_;
  float bgIntegral_; float lp_; float integral_;

  // GPU data structures

  // Covariate data is shared over a copy
  size_t* covariateCopies_;
  float* devAnimals_;
  size_t animalsPitch_;
  float* devDVal_; int* devDRowPtr_; int* devDColInd_; int* hostDRowPtr_; size_t dnnz_; //CRS
#ifdef __CUDACC__

#endif

  size_t animalsInfPowPitch_, animalsSuscPowPitch_;
  float* devAnimalsInfPow_; float* devAnimalsSuscPow_;
  float* devEventTimes_;

  size_t eventTimesPitch_;
  float* devSusceptibility_;
  float* devInfectivity_;
  thrust::device_vector<float> devProduct_;
  thrust::device_vector<float> devIntegral_;
  int integralBuffSize_;

  // Parameters
  float epsilon_;
  float gamma1_;
  float gamma2_;
  float* devXi_;
  float* devPsi_;
  float* devZeta_;
  float* devPhi_;
  float delta_;

  // GPU BLAS handles
  cublasStatus_t blasStat_;
  cublasHandle_t cudaBLAS_;
  cusparseStatus_t sparseStat_;
  cusparseHandle_t cudaSparse_;
  cusparseMatDescr_t crsDescr_;

};

#endif /* GPULIKELIHOOD_HPP_ */
