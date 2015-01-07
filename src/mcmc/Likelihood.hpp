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

#include "types.hpp"
#include "Data.hpp"
#include "PosteriorWriter.hpp"

#ifndef __CUDACC__
#include "Parameter.hpp"
#endif

#define NUMEVENTS 3

namespace EpiRisk
{

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
    void
    LoadPopulation(PopDataImporter& filename) = 0;

    virtual
    void
    LoadEpidemic(EpiDataImporter& importer) = 0;

    virtual
    void
    SetParameters(Parameter& epsilon1, Parameter& epsilon2, Parameter& gamma1,
        Parameter& gamma2, Parameters& xi, Parameters& psi, Parameters& zeta,
        Parameters& phi, Parameter& delta, Parameter& omega, Parameter& nu,
        Parameter& alpha, Parameter& a, Parameter& b) = 0;

    virtual
    void
    SetMovtBan(const float movtBanTime) = 0;

    virtual
    float
    GetMovtBan() const = 0;

    virtual const Likelihood&
    clone() = 0;

    virtual
    ~Likelihood();

    virtual
    void
    InfecCopy(const GpuLikelihood& other) = 0;

    virtual
    size_t
    GetNumKnownInfecs() const = 0;

    virtual
    size_t
    GetNumInfecs() const = 0;

    virtual
    size_t
    GetMaxInfecs() const = 0;

    virtual
    size_t
    GetNumPossibleOccults() const = 0;

    virtual
    size_t
    GetPopulationSize() const = 0;

    virtual
    void
    GetIds(std::vector<std::string>& ids) const = 0;

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
    float
    InfectionPart() = 0;

    virtual
    float
    GetIN(const size_t index) = 0;

    virtual
    float
    GetLogLikelihood() = 0;

    virtual
    float
    GetMeanI2N() const = 0;

    virtual
    float
    GetMeanOccI() const = 0;

    virtual
    void
    GetInfectiousPeriods(std::vector<EpiRisk::IPTuple_t>& periods) = 0;

    virtual friend std::ostream&
    operator<<(std::ostream& out, const Likelihood& likelihood);

  private:
    const Likelihood&
    operator=(const Likelihood& other) { return this->clone(); };

  };

} // namespace EpiRisk

#endif /* LIKELIHOOD_HPP_ */
