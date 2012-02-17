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


// CUDA defines
#define THREADSPERBLOCK 128

// Model defines
#define NUMEVENTS 3
#define NUMSPECIES 3


class GpuLikelihood
{
public:
  GpuLikelihood(const size_t realPopSize, const size_t popSize, const size_t numInfecs, const size_t nSpecies, const float obsTime, const size_t distanceNNZ);
  virtual
  ~GpuLikelihood();
  void
  SetEvents(const float* data);
  void
  SetSpecies(const float* data);
  void
  SetDistance(const float* data, const int* rowptr, const int* colind);
  void
  SetParameters(float* epsilon, float* gamma1, float* gamma2, float* xi, float* psi, float* zeta, float* phi, float* delta);
  void
  CalcEvents();
  void
  CalcSusceptibility();
  void
  CalcInfectivity();
  void
  CalcDistance();
  void
  CalcBgIntegral();
  void
  UpdateDistance();
  void
  Calculate();
  float
  LogLikelihood() const;

private:
  // Host vars
  const size_t realPopSize_;
  const size_t popSize_;
  size_t numInfecs_;
  const size_t numSpecies_;
  float logLikelihood_;
  const float obsTime_;
  float I1Time_;
  float bgIntegral_;

  // GPU data structures
  float* devAnimals_;
  size_t animalsPitch_;
  float* devAnimalsInfPow_; float* devAnimalsSuscPow_;
  size_t animalsInfPowPitch_, animalsSuscPowPitch_;
  float* devEventTimes_;
  size_t eventTimesPitch_;
  float* devSusceptibility_;
  float* devInfectivity_;
  float* devDVal_; int* devDRowPtr_; int* devDColInd_; size_t dnnz_; //CRS
  float* devTVal_;  //CRS
  float* devDTVal_; // CRS
  float* devEVal_; int* devEColPtr_; int* devERowInd_; //CCS
  float* devTmp_;

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
