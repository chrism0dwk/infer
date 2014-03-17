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

#include "types.hpp"
#include "Data.hpp"
#include "PosteriorWriter.hpp"



#ifndef __CUDACC__
#include "Parameter.hpp"
#endif

// CUDA defines
#define THREADSPERBLOCK 128

// Model defines
#define NUMEVENTS 3
#define TICKLEVELS 3
//#define NUMSPECIES 3


namespace EpiRisk
{

void
__checkCudaError(const cudaError_t err, const char* file, const int line);
#define checkCudaError(err) __checkCudaError(err, __FILE__, __LINE__)


// Data structures

  struct CsrMatrix
  {
    int* rowPtr;
    int* colInd;
    float* val;
    float* valtr;
    int nnz;
    int n;
    int m;
  };


  struct InfecIdx_t
  {
    unsigned int ptr;
    int dc;
    InfecIdx_t(const unsigned int Ptr, const int DC=-1)
    {
      ptr = Ptr;
      dc = DC;
    }
    InfecIdx_t() : ptr(NULL), dc(-1)
    {
    }
  };


// Helper classes
  template<typename T>
    class PointerVector
    {
    public:
      PointerVector()
      {
      }
      ;

      PointerVector(const size_t size)
      {
        content_.resize(size);
      }

      PointerVector(PointerVector& other)
      {
        content_ = other.content_;
      }

      const PointerVector&
      operator=(const PointerVector& other)
      {
        content_ = other.content_;
        return *this;
      }

      void
      push_back(T* x)
      {
        content_.push_back(x);
      }

      T
      operator[](const size_t index) const
      {
        return *(content_[index]);
      }
      ;

      size_t
      size() const
      {
        return content_.size();
      }
      ;

      void
      clear()
      {
        content_.clear();
      }
      ;

    private:
      std::vector<T*> content_;
    };


  class GpuLikelihood
  {
  public:
    struct LikelihoodComponents
    {
      float sumI;
      float bgIntegral;
      float logProduct;
      float integral;
    };

    explicit
    GpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic, ContactDataImporter& contact,
        const size_t nSpecies,
		  const float obsTime, const float dLimit, const bool occultsOnlyDC = false, const int gpuId=0);
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
    LoadContact(ContactDataImporter& importer);
    void
    SortPopulation();
    void
    LoadDistanceMatrix(DistMatrixImporter& filename);
    void
    CalcDistanceMatrix(const float dLimit);
    void
    SetEvents();
    void
    SetSpecies();
    void
    SetDistance(const float* data, const int* rowptr, const int* colind);
    void
    SetParameters(Parameter& epsilon1, 
		  Parameter& gamma1,
		  Parameters& phi,
		  Parameter& delta, 
		  Parameter& omega, 
		  Parameter& beta1,
		  Parameter& beta2,
		  Parameter& nu, 
		  Parameter& alpha1,
		  Parameter& alpha2, 
		  Parameter& a, 
		  Parameter& b);
    void
    RefreshParameters();
    void
    SetMovtBan(const float movtBanTime);
    float
    GetMovtBan() const;
    size_t
    GetNumKnownInfecs() const;
    size_t
    GetNumInfecs() const;
    size_t
    GetMaxInfecs() const;
    size_t
    GetNumPossibleOccults() const;
    size_t
    GetPopulationSize() const;
    void
    GetIds(std::vector<std::string>& ids) const;
    size_t
    GetNumOccults() const;
    // void
    // CalcSusceptibilityPow();
    // void
    // CalcSusceptibility();
    // void
    // CalcInfectivityPow();
    // void
    // CalcInfectivity();
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
    CalcSusceptibility();
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
    const LikelihoodComponents*
    GetLikelihoodComponents() const
    {
      return hostComponents_;
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
    void 
    GetInfectionTimes(std::vector<EpiRisk::IPTuple_t>& times);

    friend std::ostream&
    operator<<(std::ostream& out, const GpuLikelihood& likelihood);

    void
    PrintProdVector() const;
    void
    PrintLikelihoodComponents() const;
    void
    PrintParameters() const;
    void
    PrintEventTimes() const;
    void
    PrintDistMatrix() const;
    void CheckSuscep() const;
    void DumpAnimals() const;

  private:

    // Helper methods
    void
    ReduceProductVector();

    // Data import
    enum DiseaseStatus
    {
      IP = 0, DC = 1, SUSC = 2
    };

    struct Covars
    {
      string id;
      float x,y;
      DiseaseStatus status;
      float I;
      float N;
      float R;
      float ticks;
    };

    map<string, size_t> idMap_;
    typedef std::vector<Covars> Population;

    struct CompareByStatus
    {
      bool
      operator()(const Covars& lhs, const Covars& rhs) const
      {
        return (int) lhs.status < (int) rhs.status;
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
    size_t occultsOnlyDC_;

    thrust::host_vector<InfecIdx_t>* hostInfecIdx_;
    thrust::device_vector<InfecIdx_t>* devInfecIdx_;
    thrust::host_vector<InfecIdx_t>* hostSuscOccults_;
    const size_t numSpecies_;
    float logLikelihood_;
    const float obsTime_;
    float movtBan_;
    float I1Time_;
    unsigned int I1Idx_;

    LikelihoodComponents* hostComponents_;
    LikelihoodComponents* devComponents_;

    // GPU data structures

    // Covariate data is shared over a copy
    size_t* covariateCopies_;
    int* devAnimals_;
    size_t animalsPitch_;

    CsrMatrix* devD_;
    CsrMatrix* devC_;  // Rows are contactors, cols are contactees

    int* hostDRowPtr_;
    int* hostCRowPtr_;
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

    // Parameters
    float* epsilon1_;
    float* gamma1_;
    float* delta_;
    float* omega_;
    float* beta1_;
    float* beta2_;
    float* nu_;
    float* alpha1_;
    float* alpha2_;
    float* a_;
    float* b_;

    PointerVector<float> phi_;
    float* devPhi_;
    float* devHIntegCache_;

    // GPU BLAS handles
    cublasStatus_t blasStat_;
    cublasHandle_t cudaBLAS_;
    cusparseStatus_t sparseStat_;
    cusparseHandle_t cudaSparse_;
    cusparseMatDescr_t crsDescr_;

  };

  std::ostream&
  operator<<(std::ostream& out, const GpuLikelihood& likelihood);

} // namespace EpiRisk

#endif /* GPULIKELIHOOD_HPP_ */
