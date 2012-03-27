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
    : likelihood_(&likelihood)
  {
    proposal_ = new GpuLikelihood(*likelihood_);

    likelihood_->FullCalculate();
    proposal_->FullCalculate();
  }

  void
  McmcLikelihood::Accept()
  {
    *likelihood_ = *proposal_;
  }

  float
  McmcLikelihood::AddI(size_t idx, float inTime)
  {
    proposal_->AddInfectionTime(idx, inTime);
    return proposal_->GetLogLikelihood();
  }

  float
  McmcLikelihood::DeleteI(size_t idx)
  {
    proposal_->DeleteInfectionTime(idx);
    return proposal_->GetLogLikelihood();
  }

  float
  McmcLikelihood::GetCurrentValue() const
  {
    return likelihood_->GetLogLikelihood();
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
    proposal_->FullCalculate();
    return proposal_->GetLogLikelihood();
  }

  void
  McmcLikelihood::Reject()
  {
    *proposal_ = *likelihood_;
  }

  float
  McmcLikelihood::UpdateI(size_t idx, float inTime)
  {
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
} /* namespace EpiRisk */
