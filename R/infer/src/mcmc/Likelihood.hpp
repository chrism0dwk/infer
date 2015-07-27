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

#ifndef LIKELIHOOD_HPP_
#define LIKELIHOOD_HPP_

#include <map>
#include <ostream>
#include <vector>
#include <string>
#include <sys/time.h>

#include "types.hpp"
#include "Data.hpp"
#include "PosteriorWriter.hpp"
#include "DistanceMatrix.hpp"

#ifndef __CUDACC__
#include "Parameter.hpp"
#endif

#define NUMEVENTS 3

namespace EpiRisk
{

  using namespace boost::numeric;

// Constants
  const float UNITY = 1.0;
  const float ZERO = 0.0;

// Data structures

  float
  timeinseconds(const timeval a, const timeval b);

  struct InfecIdx_t
  {
    unsigned int ptr;
    int dc;
    InfecIdx_t(const unsigned int Ptr, const int DC = -1)
    {
      ptr = Ptr;
      dc = DC;
    }
    InfecIdx_t() :
        ptr(NULL), dc(-1)
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

      PointerVector(const PointerVector& other)
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

  class Likelihood
  {
  public:
    struct LikelihoodComponents
    {
      float sumI;
      float bgIntegral;
      float logProduct;
      float integral;
    };

    virtual
    ~Likelihood() {};

    void
    SetMovtBan(const float movtBanTime);

    float
    GetMovtBan() const;

    void
    SetParameters(Parameter& epsilon1, Parameter& epsilon2, Parameter& gamma1,
		  Parameter& gamma2, Parameters& xi, Parameters& psi, Parameters& zeta,
		  Parameters& phi, Parameter& delta, Parameter& omega, Parameter& nu,
		  Parameter& alpha, Parameter& a, Parameter& b);
      
    virtual
    Likelihood*
    clone() const = 0;

    const Likelihood&
    operator=(const Likelihood& other);

    virtual
    void
    InfecCopy(const Likelihood& other) = 0;

    size_t
    GetNumKnownInfecs() const;

    virtual
    size_t
    GetNumInfecs() const = 0;

    size_t
    GetMaxInfecs() const;

    virtual
    size_t
    GetNumPossibleOccults() const = 0;

    size_t
    GetPopulationSize() const;

    virtual
    void
    GetIds(std::vector<std::string>& ids) const;

    virtual
    size_t
    GetNumOccults() const = 0;

    virtual
    void
    UpdateInfectionTime(const unsigned int idx, const float inTime) = 0;

    virtual
    void
    AddInfectionTime(const unsigned int idx, const float inTime) = 0;

    virtual
    void
    DeleteInfectionTime(const unsigned int idx) = 0;

    virtual
    void
    FullCalculate() = 0;

    virtual
    void
    Calculate() = 0;

    virtual
    fp_t
    NonCentreInfecTimes(const fp_t oldGamma, const fp_t newGamma, 
			const fp_t prob) = 0;

    virtual
    fp_t
    InfectionPart() = 0;

    virtual
    float
    GetIN(const size_t index) = 0;

    virtual
    float
    GetLogLikelihood() const = 0;

    virtual
    LikelihoodComponents
    GetLikelihoodComponents() const = 0;

    virtual
    float
    GetMeanI2N() const = 0;

    virtual
    float
    GetMeanOccI() const = 0;

    virtual
    void
    GetInfectiousPeriods(std::vector<EpiRisk::IPTuple_t>& periods) = 0;

    virtual
    void
    GetSumInfectivityPow(fp_t* result) const = 0;

    virtual
    void
    GetSumSusceptibilityPow(fp_t* result) const = 0;

    virtual
    void
    PrintLikelihoodComponents() const = 0;

  protected:
    Likelihood(PopDataImporter& population, EpiDataImporter& epidemic, 
	       const size_t nSpecies, const fp_t obsTime, 
	       const bool occultsOnlyDC = false);
    Likelihood(const Likelihood& rhs);

    virtual
    const Likelihood&
    assign(const Likelihood& rhs) = 0;

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
    };

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

    // Data members
    map<string, size_t> idMap_;
    typedef std::vector<Covars> Population;
    Population population_;

    const size_t popSize_;
    const size_t numSpecies_;
    const size_t numKnownInfecs_;
    const fp_t obsTime_;
    const fp_t movtBan_;
    const size_t maxInfecs_;
    const size_t occultsOnlyDC_;

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

  private:
    void
    LoadPopulation(PopDataImporter& importer);
    void
    LoadEpidemic(EpiDataImporter& importer);
    //fp_t
    //LoadDistanceMatrix(DistMatrixImporter& importer);
    void
    SortPopulation();


  };

} // namespace EpiRisk

#endif /* LIKELIHOOD_HPP_ */
