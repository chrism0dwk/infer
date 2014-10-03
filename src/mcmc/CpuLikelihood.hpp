/*************************************************************************
 *  ./src/mcmc/CpuLikelihood.hpp
 *  Copyright Chris Jewell <c.p.jewell@massey.ac.nz> 2013
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
 *  Created on: Nov 20th, 2013
 *      Author: stsiab
 */

#ifndef CPULIKELIHOOD_HPP_
#define CPULIKELIHOOD_HPP_

#include <map>
#include <ostream>
#include <vector>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "types.hpp"
#include "Data.hpp"
#include "PosteriorWriter.hpp"



#include "Parameter.hpp"



// Model defines
#define NUMEVENTS 3
//#define NUMSPECIES 3

namespace EpiRisk
{

  using namespace boost::numeric;

// Data structures

  struct CsrMatrix
  {
    int* rowPtr;
    int* colInd;
    float* val;
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


  class CpuLikelihood
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
    CpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic,
        const size_t nSpecies,
		  const float obsTime, const float dLimit, const bool occultsOnlyDC = false);
    virtual
    ~CpuLikelihood();
    void
    LoadPopulation(PopDataImporter& filename);
    void
    LoadEpidemic(EpiDataImporter& importer);
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
		  Parameter& epsilon2, 
		  Parameter& gamma1, 
		  Parameter& gamma2,
		  Parameters& xi, 
		  Parameters& psi, 
		  Parameters& zeta, 
		  Parameters& phi,
		  Parameter& delta, 
		  Parameter& omega, 
		  Parameter& nu, 
		  Parameter& alpha, 
		  Parameter& a, 
		  Parameter& b);
    void
    SetMovtBan(const float movtBan);
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
    GetPopulationSize() const;
    void
    GetIds(std::vector<std::string>& ids) const;
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
    CalcProduct();
    void
    CalcIntegral();
    void
    FullCalculate();
    void
    Calculate();
    fp_t
    GetLogLikelihood() const;
    LikelihoodComponents
    GetLikelihoodComponents() const
    {
      return likComponents_;
    }

    void
    PrintLikelihoodComponents() const;
    void
    PrintParameters() const;
    void
    PrintEventTimes() const;
    void
    PrintProdCache() const;

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
      float cattle;
      float pigs;
      float sheep;
    } __attribute__ ((aligned (16)));

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

    Population population_;

    // Host vars
    const size_t popSize_;
    size_t popSizePitch_;
    size_t numKnownInfecs_;
    size_t maxInfecs_,maxInfecsPitch_;
    size_t occultsOnlyDC_;

    ublas::vector<InfecIdx_t> infecIdx_;
    ublas::vector<InfecIdx_t> suscOccults_;
    const size_t numSpecies_;
    float logLikelihood_;
    const float obsTime_;
    float movtBan_;
    float I1Time_;
    unsigned int I1Idx_;
    ublas::vector<fp_t> productCache_;
    LikelihoodComponents likComponents_;

    // GPU data structures

    // Covariate data is shared over a copy
    ublas::matrix<float,ublas::column_major> animals_;
    ublas::compressed_matrix<float> D_;

    size_t dnnz_; //CRS

    ublas::matrix<float,ublas::column_major> animalsInfPow_;
    ublas::matrix<float,ublas::column_major> animalsSuscPow_;
    ublas::matrix<float,ublas::column_major> eventTimes_;

    ublas::vector<float> susceptibility_;
    ublas::vector<float> infectivity_;
    ublas::vector<float> devProduct_;
    float* workspaceA_;
    float* workspaceB_;
    float* workspaceC_;

    int integralBuffSize_;

    // Parameters
    float* epsilon1_;
    float* epsilon2_;
    float* gamma1_;
    float* gamma2_;
    float* delta_;
    float* omega_;
    float* nu_;
    float* alpha_;
    float* a_;
    float* b_;

    PointerVector<float> xi_;
    PointerVector<float> psi_;
    PointerVector<float> zeta_;
    PointerVector<float> phi_;

  };


} // namespace EpiRisk

#endif /* CPULIKELIHOOD_HPP_ */
