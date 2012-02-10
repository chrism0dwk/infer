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
#include <algorithm>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>



#define ADAPTIVESCALE 1.0

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
    cout << "Candidate likelihood: " << logLikCan.global << endl;
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
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates vars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            vars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
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
      updateGroup_[p].setValue(updateGroup_[p].getValue() + vars[p]);

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p].prior());

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
          updateGroup_[p].setValue(oldParams[p]);
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
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
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
      updateGroup_[p].setValue(updateGroup_[p].getValue() * exp(logvars[p]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p].prior());

    // Proposal ratio
    double qRatio = 0.0;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      qRatio += log(updateGroup_[p].getValue() / oldParams[p]);

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
          updateGroup_[p].setValue(oldParams[p]);
      }

    ++numUpdates_;
  }

  InfectivityMRW::InfectivityMRW(const string& tag,
      UpdateBlock& params, UpdateBlock& powers, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), powers_(
        powers), burnin_(burnin)
  {

    constants_.resize(3,0.0);

    transformedGroup_.add(updateGroup_[1]);
    transformedGroup_.add(updateGroup_[2]);

    // Initialize the standard covariance
    stdCov_ = new EmpCovar<LogTransform>::CovMatrix(transformedGroup_.size());
    for (size_t i = 0; i < transformedGroup_.size(); ++i)
      {
        for (size_t j = 0; j < transformedGroup_.size(); ++j)
          {
            if (i == j)
              (*stdCov_)(i, j) = 0.01 / transformedGroup_.size();
            else
              (*stdCov_)(i, j) = 0.0;
          }
      }

    empCovar_ = new EmpCovar<LogTransform> (transformedGroup_, *stdCov_);
  }

  InfectivityMRW::~InfectivityMRW()
  {
    delete stdCov_;
    delete empCovar_;
  }

  void
  InfectivityMRW::update()
  {
    // Save parameters
    std::vector<double> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Sample posterior
    empCovar_->sample();

    // Calculate constants
    std::fill(constants_.begin(),constants_.end(),0.0);
    for(Population<TestCovars>::InfectiveIterator it = env_->pop_.infecBegin();
        it != env_->pop_.infecEnd();
        it++)
      {
        constants_[0] += it->getCovariates().cattleinf;
        constants_[1] += it->getCovariates().pigsinf;
        constants_[2] += it->getCovariates().sheepinf;
      }

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0].getValue() * constants_[0];
    transform(1) = updateGroup_[0].getValue() * updateGroup_[1].getValue() * constants_[1];
    transform(2) = updateGroup_[0].getValue() * updateGroup_[2].getValue() * constants_[2];

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
                / transformedGroup_.size());
          }
        catch (cholesky_error& e)
          {
            logvars = random_.mvgauss(*stdCov_);
          }
      }
    else
      logvars = random_.mvgauss(*stdCov_);

    // Use indep gaussians here
    transform(1) *= exp(logvars(0)); //exp(random_.gaussian(0,0.8));
    transform(2) *= exp(logvars(1)); //exp(random_.gaussian(0,0.1));
    transform(0) = R - transform(1) - transform(2);

    // Transform back
    updateGroup_[0].setValue(transform(0) / constants_[0]);
    updateGroup_[1].setValue(transform(1) / (updateGroup_[0].getValue() * constants_[1]));
    updateGroup_[2].setValue(transform(2) / (updateGroup_[0].getValue() * constants_[2]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

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
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
      }

    ++numUpdates_;

  }

  InfectivityMRW::Covariance
  InfectivityMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  SusceptibilityMRW::SusceptibilityMRW(const string& tag,
      UpdateBlock& params, UpdateBlock& powers, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), powers_(
        powers), burnin_(burnin)
  {

    constants_.resize(3,0.0);

    transformedGroup_.add(updateGroup_[1]);
    transformedGroup_.add(updateGroup_[2]);

    // Initialize the standard covariance
    stdCov_ = new EmpCovar<LogTransform>::CovMatrix(transformedGroup_.size());
    for (size_t i = 0; i < transformedGroup_.size(); ++i)
      {
        for (size_t j = 0; j < transformedGroup_.size(); ++j)
          {
            if (i == j)
              (*stdCov_)(i, j) = 0.01 / transformedGroup_.size();
            else
              (*stdCov_)(i, j) = 0.0;
          }
      }

    empCovar_ = new EmpCovar<LogTransform> (transformedGroup_, *stdCov_);
  }

  SusceptibilityMRW::~SusceptibilityMRW()
  {
    delete stdCov_;
    delete empCovar_;
  }

  void
  SusceptibilityMRW::update()
  {
    // Save parameters
    std::vector<double> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Sample posterior
    empCovar_->sample();

    // Calculate constants
    std::fill(constants_.begin(),constants_.end(),0.0);
    for(Population<TestCovars>::PopulationIterator it = env_->pop_.begin();
        it != env_->pop_.end();
        ++it)
      {
        constants_[0] += it->getCovariates().cattlesusc;
        constants_[1] += it->getCovariates().pigssusc;
        constants_[2] += it->getCovariates().sheepsusc;
      }

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0].getValue() * constants_[0];
    transform(1) = updateGroup_[0].getValue() * updateGroup_[1].getValue() * constants_[1];
    transform(2) = updateGroup_[0].getValue() * updateGroup_[2].getValue() * constants_[2];

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
                / transformedGroup_.size());
          }
        catch (cholesky_error& e)
          {
            logvars = random_.mvgauss(*stdCov_);
          }
      }
    else
      logvars = random_.mvgauss(*stdCov_);

    // Use indep gaussians here
    transform(1) *= exp(logvars(0)); //exp(random_.gaussian(0,0.8));
    transform(2) *= exp(logvars(1)); //exp(random_.gaussian(0,0.1));
    transform(0) = R - transform(1) - transform(2);

    // Transform back
    updateGroup_[0].setValue(transform(0) / constants_[0]);
    updateGroup_[1].setValue(transform(1) / (updateGroup_[0].getValue() * constants_[1]));
    updateGroup_[2].setValue(transform(2) / (updateGroup_[0].getValue() * constants_[2]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

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
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
      }

    ++numUpdates_;

  }

  SusceptibilityMRW::Covariance
  SusceptibilityMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  SpeciesMRW::SpeciesMRW(const string& tag,
      UpdateBlock& params, std::vector<double>& alpha, size_t burnin, Random& rng,
      Likelihood& logLikelihood, Mcmc* env) :
    McmcUpdate(tag, rng, logLikelihood, env), updateGroup_(params), constants_(
        alpha), burnin_(burnin)
  {

    transformedGroup_.add(updateGroup_[1]);
    transformedGroup_.add(updateGroup_[2]);

    // Initialize the standard covariance
    stdCov_ = new EmpCovar<LogTransform>::CovMatrix(transformedGroup_.size());
    for (size_t i = 0; i < transformedGroup_.size(); ++i)
      {
        for (size_t j = 0; j < transformedGroup_.size(); ++j)
          {
            if (i == j)
              (*stdCov_)(i, j) = 0.01 / transformedGroup_.size();
            else
              (*stdCov_)(i, j) = 0.0;
          }
      }

    empCovar_ = new EmpCovar<LogTransform> (transformedGroup_, *stdCov_);
  }

  SpeciesMRW::~SpeciesMRW()
  {
    delete stdCov_;
    delete empCovar_;
  }

  void
  SpeciesMRW::update()
  {
    // Save parameters
    std::vector<double> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Sample posterior
    empCovar_->sample();

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0].getValue() * constants_[0];
    transform(1) = updateGroup_[0].getValue() * updateGroup_[1].getValue() * constants_[1];
    transform(2) = updateGroup_[0].getValue() * updateGroup_[2].getValue() * constants_[2];

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
                / transformedGroup_.size());
          }
        catch (cholesky_error& e)
          {
            logvars = random_.mvgauss(*stdCov_);
          }
      }
    else
      logvars = random_.mvgauss(*stdCov_);

    // Use indep gaussians here
    transform(1) *= exp(logvars(0)); //exp(random_.gaussian(0,0.8));
    transform(2) *= exp(logvars(1)); //exp(random_.gaussian(0,0.1));
    transform(0) = R - transform(1) - transform(2);

    // Transform back
    updateGroup_[0].setValue(transform(0) / constants_[0]);
    updateGroup_[1].setValue(transform(1) / (updateGroup_[0].getValue() * constants_[1]));
    updateGroup_[2].setValue(transform(2) / (updateGroup_[0].getValue() * constants_[2]));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

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
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
      }

    ++numUpdates_;

  }

  SpeciesMRW::Covariance
  SpeciesMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }


  InfectivityPowMRW::InfectivityPowMRW(const std::string& tag,
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

    // Create species inf cache
    cache_ = new ublas::matrix<double>(env->pop_.size(),params.size());

  }
  ;

  InfectivityPowMRW::~InfectivityPowMRW()
  {
    delete empCovar_;
    delete stdCov_;
    delete cache_;
  }

  void
  InfectivityPowMRW::setCovariance(
      EmpCovar<LogTransform>::CovMatrix& covariance)
  {
    // Start the empirical covariance matrix
    delete empCovar_;
    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, covariance);
  }
  InfectivityPowMRW::Covariance
  InfectivityPowMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  void
  InfectivityPowMRW::update()
  {
    // Save old values
    std::vector<double> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
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
      updateGroup_[p].setValue(updateGroup_[p].getValue() * exp(logvars[p]));

    // Update species infectivity covars
    int counter = 0;
    for(Population<TestCovars>::PopulationIterator it = env_->pop_.begin();
        it != env_->pop_.end();
        ++it,++counter)
      {
        const TestCovars& covars(it->getCovariates());
        (*cache_)(counter,0) = covars.cattleinf;
        (*cache_)(counter,1) = covars.pigsinf;
        (*cache_)(counter,2) = covars.sheepinf;
        const_cast<TestCovars&>(it->getCovariates()).cattleinf = pow(covars.cattle,updateGroup_[0].getValue());
        const_cast<TestCovars&>(it->getCovariates()).pigsinf = pow(covars.pigs,updateGroup_[1].getValue());
        const_cast<TestCovars&>(it->getCovariates()).sheepinf = pow(covars.sheep,updateGroup_[2].getValue());
      }

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p].prior());

    // Proposal ratio
    double qRatio = 0.0;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      qRatio += log(updateGroup_[p].getValue() / oldParams[p]);

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
          updateGroup_[p].setValue(oldParams[p]);
        counter = 0;
        for(Population<TestCovars>::PopulationIterator it = env_->pop_.begin();
            it != env_->pop_.end();
            it++, counter++)
          {
            const_cast<TestCovars&>(it->getCovariates()).cattleinf = (*cache_)(counter,0);
            const_cast<TestCovars&>(it->getCovariates()).pigsinf = (*cache_)(counter,1);
            const_cast<TestCovars&>(it->getCovariates()).sheepinf = (*cache_)(counter,2);
          }
      }

    ++numUpdates_;
  }



  SusceptibilityPowMRW::SusceptibilityPowMRW(const std::string& tag,
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

    // Create species inf cache
    cache_ = new ublas::matrix<double>(env->pop_.size(),params.size());

  }
  ;

  SusceptibilityPowMRW::~SusceptibilityPowMRW()
  {
    delete empCovar_;
    delete stdCov_;
    delete cache_;
  }

  void
  SusceptibilityPowMRW::setCovariance(
      EmpCovar<LogTransform>::CovMatrix& covariance)
  {
    // Start the empirical covariance matrix
    delete empCovar_;
    empCovar_ = new EmpCovar<LogTransform> (updateGroup_, covariance);
  }
  SusceptibilityPowMRW::Covariance
  SusceptibilityPowMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  void
  SusceptibilityPowMRW::update()
  {
    // Save old values
    std::vector<double> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Calculate current posterior
    double logPiCur = logLikelihood_.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * ADAPTIVESCALE
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
      updateGroup_[p].setValue(updateGroup_[p].getValue() * exp(logvars[p]));

    // Update species infectivity covars
    int counter = 0;
    for(Population<TestCovars>::PopulationIterator it = env_->pop_.begin();
        it != env_->pop_.end();
        ++it,++counter)
      {
        const TestCovars& covars(it->getCovariates());
        (*cache_)(counter,0) = covars.cattlesusc;
        (*cache_)(counter,1) = covars.pigssusc;
        (*cache_)(counter,2) = covars.sheepsusc;
        const_cast<TestCovars&>(it->getCovariates()).cattlesusc = pow(covars.cattle,updateGroup_[0].getValue());
        const_cast<TestCovars&>(it->getCovariates()).pigssusc = pow(covars.pigs,updateGroup_[1].getValue());
        const_cast<TestCovars&>(it->getCovariates()).sheepsusc = pow(covars.sheep,updateGroup_[2].getValue());
      }

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    double logPiCan = logLikCan.global;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p].prior());

    // Proposal ratio
    double qRatio = 0.0;
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      qRatio += log(updateGroup_[p].getValue() / oldParams[p]);

    // Accept or reject
    double accept = logPiCan - logPiCur + qRatio;

    cerr << "SUSCEP POW ACCEPT/REJECT = " << accept << endl;
    if (log(random_.uniform()) < accept)
      {
        logLikelihood_ = logLikCan;
        acceptance_++;
        cerr << "ACCEPT SUSCEP POW" << endl;
      }
    else
      {
        for (size_t p = 0; p < updateGroup_.size(); ++p)
          updateGroup_[p].setValue(oldParams[p]);
        counter = 0;
        for(Population<TestCovars>::PopulationIterator it = env_->pop_.begin();
            it != env_->pop_.end();
            it++, counter++)
          {
            const_cast<TestCovars&>(it->getCovariates()).cattlesusc = (*cache_)(counter,0);
            const_cast<TestCovars&>(it->getCovariates()).pigssusc = (*cache_)(counter,1);
            const_cast<TestCovars&>(it->getCovariates()).sheepsusc = (*cache_)(counter,2);
          }
        cerr << "REJECT SUSCEP POW" << endl;
      }

    ++numUpdates_;
  }



  SellkeSerializer::SellkeSerializer(const std::string filename, Random& rng, Likelihood& logLikelihood, Mcmc* const env)
    : McmcUpdate(filename, rng, logLikelihood, env)
  {

    if(env_->mpirank_ == 0) {
        outfile_.open(filename.c_str(),ios::out);
        if(!outfile_.is_open()) {
            string msg = "Cannot open SellkeSerializer output file '";
            msg += filename;
            msg += "'";
            throw output_exception(msg.c_str());
        }
    }
  }
  SellkeSerializer::~SellkeSerializer()
  {
    outfile_.close();
  }
  void
  SellkeSerializer::update()
  {
    std::vector< map<string, double> > pressures;
    //logLikelihood_.integPressure=pressures;
    //boost::mpi::gather(env_->comm_,logLikelihood_.integPressure, pressures, 0);

    if(env_->mpirank_ == 0) {
        bool isFirst = true;
        for(std::vector< map<string, double> >::const_iterator jt = pressures.begin();
            jt != pressures.end();
            jt++) {
          for(map<string,double>::const_iterator it = jt->begin();
              it != jt->end();
              it++)
            {
              if(!isFirst) outfile_ << " ";
              else isFirst = false;

              if(env_->pop_.getById(it->first).getI() != POSINF)
                outfile_ << it->first << ":0:" << it->second;
              else outfile_ << it->first << ":1:" << it->second;
            }
        }
        outfile_ << "\n";
    }
  }


}
