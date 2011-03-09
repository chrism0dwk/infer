/*************************************************************************
 *  ./src/mcmc/MCMCUpdater.cpp
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
 * MCMCUpdater.cpp
 *
 *  Created on: 21 Jan 2011
 *      Author: stsiab
 */

#include "MCMCUpdater.hpp"

namespace EpiRisk
{

  McmcUpdate::McmcUpdate(const std::string& tag, ParameterView& params, Random& rng,
      Likelihood& logLikelihood, Mcmc* const env) :
    tag_(tag), updateGroup_(params), random_(rng), logLikelihood_(logLikelihood), env_(env),
      acceptance_(0),
      numUpdates_(0)
  {
  }

  McmcUpdate::~McmcUpdate()
  {
  }

  double
  McmcUpdate::getAcceptance() const
  {
    return (double) acceptance_ / (double) numUpdates_;
  }

  std::string
  McmcUpdate::getTag() const
  {
    return tag_;
  }


  AdaptiveMultiLogMRW::AdaptiveMultiLogMRW(const std::string& tag, ParameterView& params, Random& rng,
          Likelihood& logLikelihood, Mcmc* const env ) :
          McmcUpdate(tag,params,rng,logLikelihood,env) {

    // Initialize the standard covariance
        stdCov_ = new EmpCovar<LogTransform>::CovMatrix(updateGroup_.size());
        for (size_t i = 0; i < updateGroup_.size(); ++i)
          {
            for (size_t j = 0; j < updateGroup_.size(); ++j)
              {
                if (i == j)
                  (*stdCov_)(i, j) = 0.01 / updateGroup_.size();
                else
                  (*stdCov_)(i, j) = 0.0;
              }
          }

        empCovar_ = new EmpCovar<LogTransform> (updateGroup_, *stdCov_, ADMLM_BURNIN);

  };

  AdaptiveMultiLogMRW::~AdaptiveMultiLogMRW()
  {
    delete empCovar_;
    delete stdCov_;
  }

  void
  AdaptiveMultiLogMRW::setCovariance(EmpCovar<LogTransform>::CovMatrix& covariance)
  {
    // Start the empirical covariance matrix
    delete empCovar_;
    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, covariance,5000);
  }

  void
  AdaptiveMultiLogMRW::update()
  {
    // Save old values
    std::vector<double> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = *(updateGroup_[i]);

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p]->prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * 2.38 * 2.38
                / updateGroup_.size());
          }
        catch (cholesky_error& e)
          {
            logvars = random_.mvgauss(*stdCov_);
          }
      }
    else
      logvars = random_.mvgauss(*stdCov_);

    // Log MRW proposal
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      *(updateGroup_[p]) *= exp(logvars[p]);

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p]->prior());

    // Proposal ratio
    double qRatio = 0.0;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      qRatio += log(*(updateGroup_[p]) / oldParams[p]);

    // Accept or reject
    double accept = logPiCan - logPiCur + qRatio;
    if (log(random_.uniform()) < accept)
      {
        logLikelihood_ = logLikCan;
        acceptance_++;
      }
    else
      {
        for (size_t p = 0; p < updateGroup_.size(); ++p)
          *(updateGroup_[p]) = oldParams[p];
      }

    ++numUpdates_;
  }

}
