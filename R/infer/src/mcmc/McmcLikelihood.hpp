/*************************************************************************
 *  ./src/mcmc/McmcLikelihood.hpp
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
 * McmcLikelihood.hpp
 *
 *  Created on: Mar 20, 2012
 *      Author: stsiab
 */

#ifndef MCMCLIKELIHOOD_HPP_
#define MCMCLIKELIHOOD_HPP_

#include "GpuLikelihood.hpp"



namespace EpiRisk
{
 namespace Mcmc
 {



  class LikelihoodHandler
  {
  public:
    explicit
    LikelihoodHandler(GpuLikelihood& likelihood);
    virtual
    ~LikelihoodHandler();
    float
    Propose();
    float
    UpdateI(size_t idx, float inTime);
    float
    AddI(size_t idx, float inTime);
    float
    DeleteI(size_t idx);
    float
    GetCurrentValue() const;
    float
    GetInfectionPart(const bool proposal = false) const;
    float
    GetMeanI2N() const;
    float
    GetMeanOccI() const;
    void
    GetSumInfectivityPow(float* result) const;
    void
    GetSumSusceptibilityPow(float* result) const;
    void
    Accept();
    void
    Reject();
    size_t
    GetNumInfecs() const;
    size_t
    GetNumPossibleOccults() const;
    size_t
    GetNumOccults() const;
    size_t
    GetNumKnownInfecs() const;
    bool
    IsInfecDC(const size_t index) const;
    float
    GetIN(const size_t index) const;
    float
    GetValue() const;
    float
    GetSumI() const { return likelihood_->GetSumI(); }
    float
    ProposeNCInfecTimes(const float oldGamma, const float newGamma, const float r);
    float
    GetNCInfecTimes(const float power) const;
    const GpuLikelihood::LikelihoodComponents*
    GetProposal() const { return proposal_->GetLikelihoodComponents(); }
    const GpuLikelihood::LikelihoodComponents*
    GetCurrent() const { return likelihood_->GetLikelihoodComponents(); }
    void
    CompareProdVectors() const;
  private:
    GpuLikelihood* likelihood_;
    GpuLikelihood* proposal_;
    enum move_t
    {
      PARAMETER=0,
      INFECTIME,
      ADD,
      DELETE
    } lastMove_;
  };
 }
} /* namespace EpiRisk */
#endif /* MCMCLIKELIHOOD_HPP_ */
