/*************************************************************************
 *  ./src/mcmc/McmcLikelihood.cpp
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
 * McmcLikelihood.cpp
 *
 *  Created on: Mar 20, 2012
 *      Author: stsiab
 */

#include "McmcLikelihood.hpp"

namespace EpiRisk
{

  McmcLikelihood::McmcLikelihood(GpuLikelihood& likelihood)
    : likelihood_(&likelihood), lastMove_(PARAMETER)
  {
    proposal_ = new GpuLikelihood(*likelihood_);

    likelihood_->FullCalculate();
    proposal_->FullCalculate();
  }

  McmcLikelihood::~McmcLikelihood()
  {
    delete proposal_;
  }

  void
  McmcLikelihood::Accept()
  {
    if (lastMove_ == PARAMETER)
      *likelihood_ = *proposal_;
    else
      likelihood_->InfecCopy(*proposal_);

    lastMove_ = PARAMETER;
  }

  float
  McmcLikelihood::AddI(size_t idx, float inTime)
  {
    lastMove_ = ADD;
    proposal_->AddInfectionTime(idx, inTime);
    return proposal_->GetLogLikelihood();
  }

  float
  McmcLikelihood::DeleteI(size_t idx)
  {
    lastMove_ = DELETE;
    proposal_->DeleteInfectionTime(idx);
    return proposal_->GetLogLikelihood();
  }

  float
  McmcLikelihood::GetCurrentValue() const
  {
    return likelihood_->GetLogLikelihood();
  }

  float
  McmcLikelihood::GetInfectionPart(const bool proposal) const
  {
    if(proposal) return proposal_->InfectionPart();
    else return likelihood_->InfectionPart();
  }

  float
  McmcLikelihood::GetMeanI2N() const
  {
    return likelihood_->GetMeanI2N();
  }

  float
  McmcLikelihood::GetMeanOccI() const
  {
    return likelihood_->GetMeanOccI();
  }

  void
  McmcLikelihood::GetSumInfectivityPow(float* result) const
  {
    return likelihood_->GetSumInfectivityPow(result);
  }

  void
  McmcLikelihood::GetSumSusceptibilityPow(float* result) const
  {
    return likelihood_->GetSumSusceptibilityPow(result);
  }

  float
  McmcLikelihood::Propose()
  {
    lastMove_ = PARAMETER;
    proposal_->FullCalculate();
    return proposal_->GetLogLikelihood();
  }

  void
  McmcLikelihood::Reject()
  {
    if (lastMove_ == PARAMETER)
      *proposal_ = *likelihood_;
    else
      proposal_->InfecCopy(*likelihood_);
  }

  float
  McmcLikelihood::UpdateI(size_t idx, float inTime)
  {
    lastMove_ = INFECTIME;
    proposal_->UpdateInfectionTime(idx, inTime);

    return proposal_->GetLogLikelihood();
  }

  size_t
  McmcLikelihood::GetNumInfecs() const
  {
    return likelihood_->GetNumInfecs();
  }

  float
  McmcLikelihood::GetIN(const size_t index) const
  {
    return likelihood_->GetIN(index);
  }

  size_t
  McmcLikelihood::GetNumPossibleOccults() const
  {
    return likelihood_->GetNumPossibleOccults();
  }

  size_t
  McmcLikelihood::GetNumOccults() const
  {
    return likelihood_->GetNumOccults();
  }

  size_t
  McmcLikelihood::GetNumKnownInfecs() const
  {
    return likelihood_->GetNumKnownInfecs();
  }

  bool
  McmcLikelihood::IsInfecDC(const size_t index) const
  {
    return index >= likelihood_->GetNumKnownInfecs() and index < likelihood_->GetMaxInfecs();
  }

  float
  McmcLikelihood::GetValue() const
  {
    return likelihood_->GetLogLikelihood();
  }
  float
  McmcLikelihood::NonCentreInfecTimes(const float oldGamma, const float newGamma, const float prob)
  {
    return proposal_->NonCentreInfecTimes(oldGamma, newGamma, prob);
  }
} /* namespace EpiRisk */
