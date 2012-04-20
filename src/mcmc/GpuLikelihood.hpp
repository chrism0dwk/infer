/*************************************************************************
 *  ./src/mcmc/GpuLikelihood.hpp
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

#include <map>
#include <ostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Data.hpp"

#ifndef __CUDACC__
#include "Parameter.hpp"
#else
namespace EpiRisk {
  class Parameter;
  class Parameters;
}
#endif

// CUDA defines
#define THREADSPERBLOCK 64

// Model defines
#define NUMEVENTS 3
#define NUMSPECIES 3

// Helper classes
template <typename T>
class PointerVector
{
public:
  PointerVector() {};

  PointerVector(const size_t size)
   { content_.resize(size); }

  PointerVector(PointerVector& other)
  { content_ = other.content_; }

  const PointerVector&
  operator=(const PointerVector& other)
  {
    content_ = other.content_;
    return *this;
  }

  void
  push_back(T* x)
  { content_.push_back(x); }

  T
  operator[](const size_t index) const
    { return *(content_[index]); };

  size_t
  size() const { return content_.size(); };

  void
  clear() { content_.clear(); };

private:
  std::vector<T*> content_;
};

using EpiRisk::Parameter;
using EpiRisk::Parameters;

class GpuLikelihood
{
public:
  explicit
  GpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic,
      DistMatrixImporter& distMatrix, const size_t nSpecies, const float obsTime,
      const bool occultsOnlyDC = true);
  explicit
  GpuLikelihood(const GpuLikelihood& other);
  virtual
  ~GpuLikelihood();
  const GpuLikelihood&
  operator=(const GpuLikelihood& other);
  void
  InfecCopy(const GpuLikelihood& other);
  void
  LoadPopulation(PopDataImporter& filename);
  void
  LoadEpidemic(EpiDataImporter& importer);
  void
  SortPopulation();
  void
  LoadDistanceMatrix(DistMatrixImporter& filename);
  void
  SetEvents();
  void
  SetSpecies();
  void
  SetDistance(const float* data, const int* rowptr, const int* colind);
  void
  SetParameters(Parameter& epsilon, Parameter& gamma1, Parameter& gamma2, Parameters& xi,
      Parameters& psi, Parameters& zeta, Parameters& phi, Parameter& delta, Parameter& a, Parameter& b);
  void
  RefreshParameters();
  size_t
  GetNumKnownInfecs() const;
  size_t
  GetNumInfecs() const;
  size_t
  GetMaxInfecs() const;
  size_t
  GetNumPossibleOccults() const;
  size_t
  GetNumOccults() const;
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
  float
  InfectionPart();
  float
  GetIN(const size_t index);
  float
  GetLogLikelihood() const;
  float
  GetN(const int idx) const;
  float
  GetMeanI2N() const;
  float
  GetMeanOccI() const;
  void
  GetSumInfectivityPow(float* result) const;
  void
  GetSumSusceptibilityPow(float* result) const;
  void
  LazyAddInfecTime(const int idx, const float inTime);

  friend std::ostream& operator<<(std::ostream& out, const GpuLikelihood& likelihood);

private:
  // Data import
  enum DiseaseStatus
  {
    IP = 0, DC = 1, SUSC = 2
  };

  struct Covars
  {
    string id;
    DiseaseStatus status;
    float I;
    float N;
    float R;
    float cattle;
    float pigs;
    float sheep;
  };

  map<string, size_t> idMap_;
  typedef std::vector<Covars> Population;

  struct CompareByStatus
  {
    bool
    operator()(const Covars& lhs, const Covars& rhs) const
    {
      return (int)lhs.status < (int)rhs.status;
    }
  };

  struct CompareByI
  {
    bool
    operator()(const Covars& lhs, const Covars& rhs) const
    {
      return lhs.I < rhs.I;
    }
  };

  Population hostPopulation_;

  // Host vars
  const size_t popSize_;
  size_t numKnownInfecs_;
  size_t maxInfecs_;
  thrust::host_vector<unsigned int> hostInfecIdx_;
  thrust::device_vector<unsigned int> devInfecIdx_;
  thrust::host_vector<unsigned int> hostSuscOccults_;
  const size_t numSpecies_;
  float logLikelihood_;
  const float obsTime_;
  float I1Time_;
  unsigned int I1Idx_;
  float sumI_;
  float bgIntegral_;
  float lp_;
  float integral_;

  // GPU data structures

  // Covariate data is shared over a copy
  size_t* covariateCopies_;
  float* devAnimals_;
  size_t animalsPitch_;
  float* devDVal_;
  int* devDRowPtr_;
  int* devDColInd_;
  int* hostDRowPtr_;
  size_t dnnz_; //CRS
#ifdef __CUDACC__

#endif

  size_t animalsInfPowPitch_, animalsSuscPowPitch_;
  float* devAnimalsInfPow_;
  float* devAnimalsSuscPow_;
  float* devEventTimes_;

  size_t eventTimesPitch_;
  float* devSusceptibility_;
  float* devInfectivity_;
  thrust::device_vector<float> devProduct_;
  thrust::device_vector<float> devIntegral_;
  int integralBuffSize_;

  // Parameters
  float* epsilon_;
  float* gamma1_;
  float* gamma2_;
  float* delta_;
  float* a_;
  float* b_;

  PointerVector<float> xi_;
  PointerVector<float> psi_;
  PointerVector<float> zeta_;
  PointerVector<float> phi_;

  float* devXi_;
  float* devPsi_;
  float* devZeta_;
  float* devPhi_;


  // GPU BLAS handles
  cublasStatus_t blasStat_;
  cublasHandle_t cudaBLAS_;
  cusparseStatus_t sparseStat_;
  cusparseHandle_t cudaSparse_;
  cusparseMatDescr_t crsDescr_;

};

std::ostream& operator<<(std::ostream& out, const GpuLikelihood& likelihood);

#endif /* GPULIKELIHOOD_HPP_ */
