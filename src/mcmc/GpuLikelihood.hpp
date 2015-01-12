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
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <cudpp.h>

#include <map>
#include <ostream>
#include <vector>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Likelihood.hpp"

// CUDA defines
#define THREADSPERBLOCK 128

namespace EpiRisk
{

  class GpuLikelihood : public Likelihood
  {
  public:
    explicit
    GpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic,
        const size_t nSpecies,
		  const float obsTime, const float dLimit, const bool occultsOnlyDC = false, const int gpuId=0);
    explicit
    GpuLikelihood(const GpuLikelihood& rhs);
    virtual
    ~GpuLikelihood();
    void
    InfecCopy(const Likelihood& rhs);
    GpuLikelihood*
    clone() const;

    size_t
    GetNumInfecs() const;
    size_t
    GetNumPossibleOccults() const;
    size_t
    GetNumOccults() const;

    void
    UpdateInfectionTime(const unsigned int idx, const float inTime);
    void
    AddInfectionTime(const unsigned int idx, const float inTime);
    void
    DeleteInfectionTime(const unsigned int idx);

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
    LikelihoodComponents
    GetLikelihoodComponents() const
    {
      return *hostComponents_;
    }
    const thrust::device_vector<float>&
    GetProdVector() const
    {
      return *devProduct_;
    }
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
    float
    NonCentreInfecTimes(const float oldGamma, const float newGamma,
        const float prob);
    void
    GetInfectiousPeriods(std::vector<EpiRisk::IPTuple_t>& periods);

    // friend std::ostream&
    // operator<<(std::ostream& out, const GpuLikelihood& likelihood);

    void
    PrintLikelihoodComponents() const;
    void
    PrintParameters() const;
    void
    PrintEventTimes() const;
    void
    PrintDistMatrix() const;

  private:

    // Helper methods
    void
    ReduceProductVector();
    void
    SetEvents();
    void
    SetSpecies();
    void
    CalcDistanceMatrix(const float dLimit);
    void
    SetDistance(const float* data, const int* rowptr, const int* colind);
    void
    RefreshParameters();
    const Likelihood&
    assign(const Likelihood& rhs);

    // Private calculation methods
    void
    CalcProduct();
    void
    CalcIntegral();
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

    thrust::host_vector<InfecIdx_t>* hostInfecIdx_;
    thrust::device_vector<InfecIdx_t>* devInfecIdx_;
    thrust::host_vector<InfecIdx_t>* hostSuscOccults_;
    float logLikelihood_;
    float I1Time_;
    unsigned int I1Idx_;

    LikelihoodComponents* hostComponents_;
    LikelihoodComponents* devComponents_;

    // GPU data structures

    // Covariate data is shared over a copy
    size_t* covariateCopies_;
    float* devAnimals_;
    size_t animalsPitch_;

    CsrMatrix* devD_;

    int* hostDRowPtr_;
    size_t dnnz_; //CRS
    curandGenerator_t cuRand_;

    size_t animalsInfPowPitch_, animalsSuscPowPitch_;
    float* devAnimalsInfPow_;
    float* devAnimalsSuscPow_;
    float* devEventTimes_;

    size_t eventTimesPitch_;
    float* devSusceptibility_;
    float* devInfectivity_;
    thrust::device_vector<float>* devProduct_;
    thrust::device_vector<float>* devWorkspace_;
    int integralBuffSize_;

    // CUDAPP bits and pieces
    CUDPPHandle cudpp_;
    CUDPPHandle addReduce_;
    CUDPPConfiguration addReduceCfg_;
    CUDPPConfiguration logAddReduceCfg_;
    CUDPPHandle minReduce_;
    CUDPPConfiguration minReduceCfg_;

    // Device parameter vectors
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

  // std::ostream&
  // operator<<(std::ostream& out, const GpuLikelihood& likelihood);

} // namespace EpiRisk

#endif /* GPULIKELIHOOD_HPP_ */
