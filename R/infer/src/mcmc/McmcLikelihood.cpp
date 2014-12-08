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
 namespace Mcmc
 {
  LikelihoodHandler::LikelihoodHandler(GpuLikelihood& likelihood)
    : likelihood_(&likelihood), lastMove_(PARAMETER)
  {
    proposal_ = new GpuLikelihood(*likelihood_);

    likelihood_->FullCalculate();
    proposal_->FullCalculate();
  }

  LikelihoodHandler::~LikelihoodHandler()
  {
    delete proposal_;
  }

  void
  LikelihoodHandler::Accept()
  {
    if (lastMove_ == PARAMETER)
      *likelihood_ = *proposal_;
    else
      likelihood_->InfecCopy(*proposal_);

    lastMove_ = PARAMETER;
  }

  float
  LikelihoodHandler::AddI(size_t idx, float inTime)
  {
    lastMove_ = ADD;
    proposal_->AddInfectionTime(idx, inTime);

    return proposal_->GetLogLikelihood();
  }

  float
  LikelihoodHandler::DeleteI(size_t idx)
  {
    lastMove_ = DELETE;
    proposal_->DeleteInfectionTime(idx);
    return proposal_->GetLogLikelihood();
  }

  float
  LikelihoodHandler::GetCurrentValue() const
  {
    return likelihood_->GetLogLikelihood();
  }

  float
  LikelihoodHandler::GetInfectionPart(const bool proposal) const
  {
    if(proposal) return proposal_->InfectionPart();
    else return likelihood_->InfectionPart();
  }

  float
  LikelihoodHandler::GetMeanI2N() const
  {
    return likelihood_->GetMeanI2N();
  }

  float
  LikelihoodHandler::GetMeanOccI() const
  {
    return likelihood_->GetMeanOccI();
  }

  void
  LikelihoodHandler::GetSumInfectivityPow(float* result) const
  {
    return likelihood_->GetSumInfectivityPow(result);
  }

  void
  LikelihoodHandler::GetSumSusceptibilityPow(float* result) const
  {
    return likelihood_->GetSumSusceptibilityPow(result);
  }

  float
  LikelihoodHandler::Propose()
  {
    lastMove_ = PARAMETER;
    proposal_->FullCalculate();

    return proposal_->GetLogLikelihood();
  }

  void
  LikelihoodHandler::Reject()
  {
    if (lastMove_ == PARAMETER)
      *proposal_ = *likelihood_;
    else
      proposal_->InfecCopy(*likelihood_);
  }

  float
  LikelihoodHandler::UpdateI(size_t idx, float inTime)
  {
    lastMove_ = INFECTIME;
    proposal_->UpdateInfectionTime(idx, inTime);

    return proposal_->GetLogLikelihood();
  }

  size_t
  LikelihoodHandler::GetNumInfecs() const
  {
    return likelihood_->GetNumInfecs();
  }

  float
  LikelihoodHandler::GetIN(const size_t index) const
  {
    return likelihood_->GetIN(index);
  }

  size_t
  LikelihoodHandler::GetNumPossibleOccults() const
  {
    return likelihood_->GetNumPossibleOccults();
  }

  size_t
  LikelihoodHandler::GetNumOccults() const
  {
    return likelihood_->GetNumOccults();
  }

  size_t
  LikelihoodHandler::GetNumKnownInfecs() const
  {
    return likelihood_->GetNumKnownInfecs();
  }

  bool
  LikelihoodHandler::IsInfecDC(const size_t index) const
  {
    return index >= likelihood_->GetNumKnownInfecs() and index < likelihood_->GetMaxInfecs();
  }

  float
  LikelihoodHandler::GetValue() const
  {
    return likelihood_->GetLogLikelihood();
  }
  float
  LikelihoodHandler::NonCentreInfecTimes(const float oldGamma, const float newGamma, const float prob)
  {
    return proposal_->NonCentreInfecTimes(oldGamma, newGamma, prob);
  }
  void
  LikelihoodHandler::CompareProdVectors() const
  {
    thrust::host_vector<float> current = likelihood_->GetProdVector();
    thrust::host_vector<float> proposal = proposal_->GetProdVector();
    cerr << "Checking prod vector (" << current.size() << "):" << endl;
    cerr.precision(15);
    for(size_t i = 0; i<current.size(); ++i)
      {
        float curr = current[i];
        float prop = proposal[i];
        if (fabs((curr - prop)/curr) > 1e-6)
          {
            cerr << i << ":\t" << curr << "\t" << prop << "\t" << curr - prop << endl;
          }
      }
  }
 }
} /* namespace EpiRisk */
