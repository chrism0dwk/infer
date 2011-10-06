/*************************************************************************
 *  ./release/brandenburg-0.3alpha/src/mcmc/MCMCUpdater.cpp
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

  McmcUpdate::McmcUpdate(const std::string& tag, Random& rng,
      Likelihood& logLikelihood, Mcmc* const env) :
    tag_(tag), random_(rng), logLikelihood_(logLikelihood), env_(env),
        acceptance_(0), numUpdates_(0)
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

  SingleSiteLogMRW::SingleSiteLogMRW(const std::string& tag, Parameter& param,
      const double tuning, Random& rng, Likelihood& logLikelihood,
      Mcmc* const env) :
    McmcUpdate(tag, rng, logLikelihood, env), param_(param), tuning_(tuning)
  {
  }

  SingleSiteLogMRW::~SingleSiteLogMRW()
  {
  }

  void
  SingleSiteLogMRW::update()
  {
    double oldValue = param_;

    // Calculate current posterior
    double logPiCur = logLikelihood_.global + log(param_.prior());

    // Proposal via log random walk
    param_ *= exp(random_.gaussian(0, tuning_));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    // Candidate posterior
    double logPiCan = logLikCan.global + log(param_.prior());

    // q-ratio
    double qratio = param_ / oldValue;

    // Accept or reject
    if (log(random_.uniform()) < logPiCan - logPiCur + qratio)
      {
        logLikelihood_ = logLikCan;
        acceptance_++;
      }
    else
      {
        param_ = oldValue;
      }

    numUpdates_++;

  }

  AdaptiveMultiMRW::AdaptiveMultiMRW(const std::string& tag,
      UpdateBlock& params, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* const env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), burnin_(
        burnin)
  {

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

    empCovar_ = new EmpCovar<Identity> (updateGroup_, *stdCov_);

  }
  ;

  AdaptiveMultiMRW::~AdaptiveMultiMRW()
  {
    delete empCovar_;
    delete stdCov_;
  }

  void
  AdaptiveMultiMRW::setCovariance(EmpCovar<Identity>::CovMatrix& covariance)
  {
    // Start the empirical covariance matrix
    delete empCovar_;
    empCovar_ = new EmpCovar<Identity> (updateGroup_, covariance);
  }
  AdaptiveMultiMRW::Covariance
  AdaptiveMultiMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  void
  AdaptiveMultiMRW::update()
  {
    // Save old values
    std::vector<double> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i]->getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p]->prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates vars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            vars = random_.mvgauss(empCovar_->getCovariance() * 5.6644
                / updateGroup_.size());
          }
        catch (cholesky_error& e)
          {
            vars = random_.mvgauss(*stdCov_);
          }
      }
    else
      vars = random_.mvgauss(*stdCov_);

    // Log MRW proposal
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      updateGroup_[p]->setValue(updateGroup_[p]->getValue() + vars[p]);

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p]->prior());

    // Proposal ratio
    double qRatio = 0.0; // Gaussian proposal cancels

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
          updateGroup_[p]->setValue(oldParams[p]);
      }

    ++numUpdates_;
  }

  AdaptiveMultiLogMRW::AdaptiveMultiLogMRW(const std::string& tag,
      UpdateBlock& params, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* const env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), burnin_(
        burnin)
  {

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

    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, *stdCov_);

  }
  ;

  AdaptiveMultiLogMRW::~AdaptiveMultiLogMRW()
  {
    delete empCovar_;
    delete stdCov_;
  }

  void
  AdaptiveMultiLogMRW::setCovariance(
      EmpCovar<LogTransform>::CovMatrix& covariance)
  {
    // Start the empirical covariance matrix
    delete empCovar_;
    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, covariance);
  }
  AdaptiveMultiLogMRW::Covariance
  AdaptiveMultiLogMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  void
  AdaptiveMultiLogMRW::update()
  {
    // Save old values
    std::vector<double> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i]->getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p]->prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * 5.6644
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
      updateGroup_[p]->setValue(updateGroup_[p]->getValue() * exp(logvars[p]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p]->prior());

    // Proposal ratio
    double qRatio = 0.0;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      qRatio += log(updateGroup_[p]->getValue() / oldParams[p]);

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
          updateGroup_[p]->setValue(oldParams[p]);
      }

    ++numUpdates_;
  }

  SpeciesMRW::SpeciesMRW(const string& tag,
      UpdateBlock& params, std::vector<double>& alpha, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), constants_(
        alpha), burnin_(burnin)
  {
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

    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, *stdCov_);
  }

  SpeciesMRW::~SpeciesMRW()
  {

  }

  void
  SpeciesMRW::update()
  {
    // Save parameters
    std::vector<double> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i]->getValue();

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0]->getValue()*(constants_[0] + updateGroup_[1]->getValue()*constants_[1] + updateGroup_[2]->getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.global
        + log(updateGroup_[0]->prior())
        + log(updateGroup_[1]->prior())
        + log(updateGroup_[2]->prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0]->getValue() * constants_[0];
    transform(1) = updateGroup_[0]->getValue() * updateGroup_[1]->getValue() * constants_[1];
    transform(2) = updateGroup_[0]->getValue() * updateGroup_[2]->getValue() * constants_[2];

    // Propose as in Haario, Sachs, Tamminen (2001)
//    Random::Variates logvars;
//    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
//      {
//        try
//          {
//            logvars = random_.mvgauss(empCovar_->getCovariance() * 5.6644
//                / updateGroup_.size());
//          }
//        catch (cholesky_error& e)
//          {
//            logvars = random_.mvgauss(*stdCov_);
//          }
//      }
//    else
//      logvars = random_.mvgauss(*stdCov_);

    // Use indep gaussians here
    transform(1) *= exp(random_.gaussian(0,0.8));
    transform(2) *= exp(random_.gaussian(0,0.1));
    transform(0) = R - transform(1) - transform(2);

    // Transform back
    updateGroup_[0]->setValue(transform(0) / constants_[0]);
    updateGroup_[1]->setValue(transform(1) / (updateGroup_[0]->getValue() * constants_[1]));
    updateGroup_[2]->setValue(transform(2) / (updateGroup_[0]->getValue() * constants_[2]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global
        + log(updateGroup_[0]->prior())
        + log(updateGroup_[1]->prior())
        + log(updateGroup_[2]->prior());

    // q-Ratio
    double qRatio = log(transform(1) / (oldParams[0]*oldParams[1]*constants_[1])) + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

    // Accept/reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qRatio)
      {
        logLikelihood_ = logLikCan;
        acceptance_++;
      }
    else
      {
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i]->setValue(oldParams[i]);
      }

    ++numUpdates_;

  }

}
