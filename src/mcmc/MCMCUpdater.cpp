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


  SingleSiteMRW::SingleSiteMRW(const std::string& tag, Parameter& param, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* const env ) :
      McmcUpdate(tag, rng, logLikelihood, env), param_(param), tuning_(tuning)
  {
  }

  SingleSiteMRW::~SingleSiteMRW()
  {
  }

  void
  SingleSiteMRW::update()
  {
    double oldValue = param_;

    // Calculate current posterior
    double logPiCur = logLikelihood_.global + log(param_.prior());

    // Proposal via log random walk
    param_ = random_.gaussian(param_,tuning_);

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    // Candidate posterior
    double logPiCan = logLikCan.global + log(param_.prior());

    // q-ratio
    double qratio = 0.0; // Gaussian proposals cancel

    cout << param_.getTag() << ": " << logPiCan << ", " << logPiCur << "; ";

    // Accept or reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qratio)
      {
        logLikelihood_ = logLikCan;
        acceptance_++;
        cout << "ACCEPT" << endl;
      }
    else
      {
        param_ = oldValue;
        cout << "REJECT" << endl;
      }

    numUpdates_++;

  }



  SingleSiteLogMRW::SingleSiteLogMRW(const std::string& tag, Parameter& param, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* const env ) :
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
    param_ *= exp(random_.gaussian(0,tuning_));

    // Calculate candidate posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);

    // Candidate posterior
    double logPiCan = logLikCan.global + log(param_.prior());

    // q-ratio
    double qratio = param_ / oldValue;

    // Accept or reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qratio && param_ > 0.0)
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


  AdaptiveMultiMRW::AdaptiveMultiMRW(const std::string& tag, UpdateBlock& params, size_t burnin, Random& rng,
          Likelihood& logLikelihood, Mcmc* const env ) :
          McmcUpdate(tag,rng,logLikelihood,env), updateGroup_(params), burnin_(burnin) {

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

  };

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



  AdaptiveMultiLogMRW::AdaptiveMultiLogMRW(const std::string& tag, UpdateBlock& params, size_t burnin, Random& rng,
          Likelihood& logLikelihood, Mcmc* const env ) :
          McmcUpdate(tag,rng,logLikelihood,env), updateGroup_(params), burnin_(burnin) {

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


  WithinFarmBetaLogMRW::WithinFarmBetaLogMRW(Parameter& param, const double alpha, const double gamma, Population<TestCovars>& pop, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* env)
  : McmcUpdate(param.getTag(),rng,logLikelihood,env), param_(param), alpha_(alpha), gamma_(gamma),tuning_(tuning), pop_(pop)
  {
    // Simulate on farm epidemics
    for ( Population<TestCovars>::PopulationIterator it = pop_.begin();
          it != pop_.end();
          it++)
      {
        it->getCovariates().epi->simulate(param_,alpha_, gamma_);
      }
  }

  WithinFarmBetaLogMRW::~WithinFarmBetaLogMRW()
  {
  }

  void
  WithinFarmBetaLogMRW::update()
  {
    double oldParam = param_;

    // Current conditional posterior
    double logPiCur = logLikelihood_.global + log(param_.prior());

    // Propose new beta
    param_ = oldParam * exp(random_.gaussian(0.0,tuning_));

    // Resimulate on farm epidemics
    for ( Population<TestCovars>::PopulationIterator it = pop_.begin();
          it != pop_.end();
          it++)
      {
        it->getCovariates().epi->simulate(param_,alpha_, gamma_);
      }

    // Calculate candidate conditional posterior
    Likelihood logLikCan;
    env_->calcLogLikelihood(logLikCan);
    double logPiCan = logLikCan.global + log(param_.prior());

    // q ratio
    double qRatio = log(param_ / oldParam);

    // Accept/reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qRatio)
      {
       logLikelihood_ = logLikCan;
       acceptance_++;
      }
    else
      {
        // Roll back to old parameter
        param_ = oldParam;

        for ( Population<TestCovars>::PopulationIterator it = pop_.begin();
              it != pop_.end();
              it++)
          {
            it->getCovariates().epi->simulate(param_,alpha_, gamma_);
          }
      }

    numUpdates_++;
  }


}
