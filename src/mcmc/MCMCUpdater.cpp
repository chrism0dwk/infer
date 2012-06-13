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
#include <boost/numeric/ublas/lu.hpp>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>



#define ADAPTIVESCALE 1.0
#define WINDOWSIZE 100
#define TUNEIN 2.5

#define INFECPROP_A 2.0
#define INFECPROP_B 0.15


namespace EpiRisk
{

  bool
  _checkNotLessThanZero(const UpdateBlock& parms)
  {
    int lessThanZero = 0;
    for(size_t i=0; i<parms.size(); ++i)
      if(parms[i].getValue() < 0.0) lessThanZero++;

    if(lessThanZero == 0) return true;
    else return false;
  }

  int 
  _determinant_sign(const ublas::permutation_matrix<std::size_t>& pm)
  {
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
      if (i != pm(i))
	pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
  }

  template<class T>  
  double 
  _determinant( const T& mat ) {
    ublas::matrix<double> m = mat;
    ublas::permutation_matrix<size_t> pm(m.size1());
    double det = 1.0;
    if( ublas::lu_factorize(m,pm) ) {
      det = 0.0;
    } else {
      for(int i = 0; i < m.size1(); i++)
	det *= m(i,i); // multiply by elements on diagonal
      det = det * _determinant_sign( pm );
    }
    return det;
  }

  inline
  double
  extremepdf(const double x, const double a, const double b)
  {
    return a * b * exp(a + b * x - a * exp(b * x));
  }

  inline
  double
  extremecdf(const double x, const double a, const double b)
  {
    return 1 - exp(-a * (exp(b * x) - 1));
  }

  inline
  double
  gammacdf(const double x, const double a, const double b)
  {
    return gsl_cdf_gamma_P(x, a, 1.0 / b);
  }

  inline
  double
  gammapdf(const double x, const double a, const double b)
  {
    return gsl_ran_gamma_pdf(x, a, 1.0 / b);
  }

  inline
  double
  gaussianTailPdf(const double x, const double mean, const double var)
  {
    return gsl_ran_gaussian_tail_pdf(x - mean, -mean, sqrt(var));
  }

  McmcUpdate::McmcUpdate(const std::string& tag, Random& random, McmcLikelihood& logLikelihood) :
    tag_(tag), random_(random), logLikelihood_(logLikelihood), acceptance_(0), numUpdates_(0)
  {
  }

  McmcUpdate::~McmcUpdate()
  {
  }

  std::map<std::string, float>
  McmcUpdate::GetAcceptance() const
  {
    std::map<std::string, float> accepts;
    accepts.insert(std::make_pair(tag_, (float) acceptance_ / (float) numUpdates_));
    return accepts;
  }

  void
  McmcUpdate::ResetAcceptance()
  {
    acceptance_ = 0;
    numUpdates_ = 0;
  }

  std::string
  McmcUpdate::GetTag() const
  {
    return tag_;
  }

  SingleSiteLogMRW::SingleSiteLogMRW(const std::string& tag, Parameter& param,
      const double tuning, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), param_(param), tuning_(tuning)
  {
  }

  SingleSiteLogMRW::~SingleSiteLogMRW()
  {
  }

  void
  SingleSiteLogMRW::Update()
  {
    double oldValue = param_;

    // Calculate current posterior
    double logPiCur =  logLikelihood_.GetCurrentValue() + log(param_.prior());

    // Proposal via log random walk
    param_ *= exp(random_.gaussian(0, tuning_));

    // Calculate candidate posterior
    float logPiCan = logLikelihood_.Propose() + log(param_.prior());

    // q-ratio
    float qratio = param_ / oldValue;

    // Accept or reject
    if (log(random_.uniform()) < logPiCan - logPiCur + qratio)
      {
        logLikelihood_.Accept();
        acceptance_++;
      }
    else
      {
        param_ = oldValue;
        logLikelihood_.Reject();
      }

    numUpdates_++;

  }

  AdaptiveMultiMRW::AdaptiveMultiMRW(const std::string& tag,
      UpdateBlock& params, size_t burnin, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), updateGroup_(params), burnin_(burnin), adaptScalar_(ADAPTIVESCALE),windowUpdates_(0), windowAcceptance_(0)
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
  AdaptiveMultiMRW::Update()
  {
    // Save old values
    std::vector<float> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.234) adaptScalar_ *= exp(-deltan);
      else adaptScalar_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Calculate current posterior
    float logPiCur = logLikelihood_.GetCurrentValue();
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates vars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            vars = random_.mvgauss(empCovar_->getCovariance() * adaptScalar_
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
    double logPiCan = logLikelihood_.Propose();
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCan += log(updateGroup_[p].prior());

    // Proposal ratio
    double qRatio = 0.0; // Gaussian proposal cancels

    // Accept or reject
    double accept = logPiCan - logPiCur + qRatio;
    if (log(random_.uniform()) < accept)
      {
        logLikelihood_.Accept();
        acceptance_++;
	windowAcceptance_++;
      }
    else
      {
        logLikelihood_.Reject();
        for (size_t p = 0; p < updateGroup_.size(); ++p)
          updateGroup_[p].setValue(oldParams[p]);
      }

    ++numUpdates_;
    ++windowUpdates_;
  }

  AdaptiveMultiLogMRW::AdaptiveMultiLogMRW(const std::string& tag,
      UpdateBlock& params, size_t burnin, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), updateGroup_(params), burnin_(
        burnin), adaptScalar_(ADAPTIVESCALE),windowUpdates_(0), windowAcceptance_(0)
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
  AdaptiveMultiLogMRW::Update()
  {
    // Save old values
    std::vector<float> oldParams(updateGroup_.size());
    for (size_t i = 0; i < updateGroup_.size(); i++)
      oldParams[i] = updateGroup_[i].getValue();

    // Update empirical covariance
    empCovar_->sample();

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.234) adaptScalar_ *= exp(-deltan);
      else adaptScalar_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Calculate current posterior
    float logPiCur = logLikelihood_.GetCurrentValue();
    for (size_t p = 0; p < updateGroup_.size(); ++p)
      logPiCur += log(updateGroup_[p].prior());

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * adaptScalar_
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

    float logPiCan = logLikelihood_.Propose();
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
        logLikelihood_.Accept();
        acceptance_++;
	windowAcceptance_++;
      }
    else
      {
        logLikelihood_.Reject();
        for (size_t p = 0; p < updateGroup_.size(); ++p)
          updateGroup_[p].setValue(oldParams[p]);
      }

    ++numUpdates_;
    ++windowUpdates_;
  }

  InfectivityMRW::InfectivityMRW(const string& tag,
      UpdateBlock& params, size_t burnin, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), updateGroup_(params), burnin_(burnin), adaptScalar_(ADAPTIVESCALE),windowUpdates_(0), windowAcceptance_(0)
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
  InfectivityMRW::Update()
  {
    // Save parameters
    std::vector<float> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Calculate constants
    logLikelihood_.GetSumInfectivityPow(&constants_[0]);

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.GetCurrentValue()
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0].getValue() * constants_[0];
    transform(1) = updateGroup_[0].getValue() * updateGroup_[1].getValue() * constants_[1];
    transform(2) = updateGroup_[0].getValue() * updateGroup_[2].getValue() * constants_[2];

    // Sample transformed posterior
    ublas::vector<double> sample = ublas::vector_range<ublas::vector<double> >(transform, ublas::range(1,transform.size()));
    empCovar_->sample(sample);

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.234) adaptScalar_ *= exp(-deltan);
      else adaptScalar_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
	    Covariance tmp = empCovar_->getCovariance();
	    //tmp = tmp /  _determinant(tmp);
            logvars = random_.mvgauss(tmp * adaptScalar_
                / transformedGroup_.size());
          }
        catch (cholesky_error& e)
          {
	    cerr << "Cholesky error in " << __PRETTY_FUNCTION__ << ": '" << e.what() << "'" <<  endl;
            logvars = random_.mvgauss(*stdCov_);
          }
      }
    else
      logvars = random_.mvgauss(*stdCov_);

    transform(1) *= exp(logvars(0)); 
    transform(2) *= exp(logvars(1)); 
    transform(0) = R - transform(1) - transform(2);

    // Reject if we get a neg value
    if(transform(0) < 0.0f) 
      {
	for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
	++numUpdates_;
	++windowUpdates_;
	return;
      }

    // Transform back
    updateGroup_[0].setValue(transform(0) / constants_[0]);
    updateGroup_[1].setValue(transform(1) / (updateGroup_[0].getValue() * constants_[1]));
    updateGroup_[2].setValue(transform(2) / (updateGroup_[0].getValue() * constants_[2]));

     // Calculate candidate posterior
    float logPiCan = logLikelihood_.Propose()
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // q-Ratio
    float qRatio = log(transform(1) / (oldParams[0]*oldParams[1]*constants_[1])) + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

    // Accept/reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qRatio)
      {
        logLikelihood_.Accept();
        acceptance_++;
	windowAcceptance_++;
      }
    else
      {
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
        logLikelihood_.Reject();
      }

    ++numUpdates_;
    ++windowUpdates_;

  }

  InfectivityMRW::Covariance
  InfectivityMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  SusceptibilityMRW::SusceptibilityMRW(const string& tag,
      UpdateBlock& params, size_t burnin, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), updateGroup_(params), burnin_(burnin), adaptScalar_(ADAPTIVESCALE),windowUpdates_(0), windowAcceptance_(0)
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
  SusceptibilityMRW::Update()
  {
    // Save parameters
    std::vector<float> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Calculate constants
    logLikelihood_.GetSumSusceptibilityPow(&constants_[0]);

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    float R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    float logPiCur = logLikelihood_.GetCurrentValue()
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // Make proposal
    ublas::vector<double> transform(updateGroup_.size());
    transform(0) = updateGroup_[0].getValue() * constants_[0];
    transform(1) = updateGroup_[0].getValue() * updateGroup_[1].getValue() * constants_[1];
    transform(2) = updateGroup_[0].getValue() * updateGroup_[2].getValue() * constants_[2];

    // Sample transformed posterior
    ublas::vector<double> sample = ublas::vector_range<ublas::vector<double> >(transform, ublas::range(1,transform.size()));
    empCovar_->sample(sample);

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.234) adaptScalar_ *= exp(-deltan);
      else adaptScalar_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Propose as in Haario, Sachs, Tamminen (2001)
    Random::Variates logvars;
    if (random_.uniform() < 0.95 and numUpdates_ > burnin_)
      {
        try
          {
            logvars = random_.mvgauss(empCovar_->getCovariance() * adaptScalar_
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

    // Reject if we get a neg value
    if(transform(0) < 0.0f) 
      {
	for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
	++numUpdates_;
	++windowUpdates_;
	return;
      }


    // Transform back
    updateGroup_[0].setValue(transform(0) / constants_[0]);
    updateGroup_[1].setValue(transform(1) / (updateGroup_[0].getValue() * constants_[1]));
    updateGroup_[2].setValue(transform(2) / (updateGroup_[0].getValue() * constants_[2]));

    // Calculate candidate posterior
    double logPiCan = logLikelihood_.Propose()
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // q-Ratio
    double qRatio = log(transform(1) / (oldParams[0]*oldParams[1]*constants_[1])) + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

    // Accept/reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qRatio)
      {
        logLikelihood_.Accept();
        acceptance_++;
	++windowAcceptance_;
      }
    else
      {
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
        logLikelihood_.Reject();
      }

    ++numUpdates_;
    ++windowUpdates_;

  }

  SusceptibilityMRW::Covariance
  SusceptibilityMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }

  SpeciesMRW::SpeciesMRW(const string& tag,
      UpdateBlock& params, std::vector<double>& alpha, size_t burnin, Random& random, McmcLikelihood& logLikelihood) :
    McmcUpdate(tag, random, logLikelihood), updateGroup_(params), constants_(
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
  SpeciesMRW::Update()
  {
    // Save parameters
    std::vector<double> oldParams(updateGroup_.size());
    for(size_t i=0; i<updateGroup_.size(); ++i) oldParams[i] = updateGroup_[i].getValue();

    // Sample posterior
    empCovar_->sample();

    // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
    double R = updateGroup_[0].getValue()*(constants_[0] + updateGroup_[1].getValue()*constants_[1] + updateGroup_[2].getValue()*constants_[2]);

    // Current posterior
    double logPiCur = logLikelihood_.GetCurrentValue()
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

    double logPiCan = logLikelihood_.Propose()
        + log(updateGroup_[0].prior())
        + log(updateGroup_[1].prior())
        + log(updateGroup_[2].prior());

    // q-Ratio
    double qRatio = log(transform(1) / (oldParams[0]*oldParams[1]*constants_[1])) + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

    // Accept/reject
    if(log(random_.uniform()) < logPiCan - logPiCur + qRatio)
      {
        logLikelihood_.Accept();
        acceptance_++;
      }
    else
      {
        for(size_t i=0; i<updateGroup_.size(); ++i) updateGroup_[i].setValue(oldParams[i]);
        logLikelihood_.Reject();
      }

    ++numUpdates_;

  }

  SpeciesMRW::Covariance
  SpeciesMRW::getCovariance() const
  {
    return empCovar_->getCovariance();
  }


  InfectionTimeGammaCentred::InfectionTimeGammaCentred(const string& tag, Parameter& param, const float tuning, Random& random, McmcLikelihood& logLikelihood)
    : McmcUpdate(tag, random, logLikelihood), param_(param), tuning_(ADAPTIVESCALE), windowUpdates_(0), windowAcceptance_(0)
  {

  }
  InfectionTimeGammaCentred::~InfectionTimeGammaCentred()
  {

  }
  void
  InfectionTimeGammaCentred::Update()
  {
    double oldValue = param_;

    // Calculate current posterior
    double logPiCur =  logLikelihood_.GetInfectionPart() + log(param_.prior());

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.44) tuning_ *= exp(-deltan);
      else tuning_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Proposal via log random walk
    param_ *= exp(random_.gaussian(0, tuning_));

    // Calculate candidate posterior
    float logPiCan = logLikelihood_.GetInfectionPart() + log(param_.prior());

    // q-ratio
    float qratio = logf(param_ / oldValue);

    // Accept or reject
    if (log(random_.uniform()) < logPiCan - logPiCur + qratio)
      {
        acceptance_++;
        windowAcceptance_++;
      }
    else
      {
        param_ = oldValue;
      }

    numUpdates_++;
    windowUpdates_++;
  }

  InfectionTimeGammaNC::InfectionTimeGammaNC(const std::string& tag, Parameter& param, const float tuning, const float ncProp, Random& random, McmcLikelihood& logLikelihood)
   : ncProp_(ncProp), param_(param), McmcUpdate(tag, random, logLikelihood), tuning_(ADAPTIVESCALE), windowUpdates_(0), windowAcceptance_(0)
    {

    }

  InfectionTimeGammaNC::~InfectionTimeGammaNC()
  {

  }

  void
  InfectionTimeGammaNC::Update()
  {
    double oldValue = param_;

    // Calculate current posterior
    double logPiCur =  logLikelihood_.GetCurrentValue() + log(param_.prior());

    // Adapt adaptscalar
    if (windowUpdates_ % WINDOWSIZE == 0 ) {
      double accept = (double)windowAcceptance_ / (double)windowUpdates_;
      float deltan = min(0.5, 1.0 / sqrtf(numUpdates_/WINDOWSIZE));
      if(accept < 0.44) tuning_ *= exp(-deltan);
      else tuning_ *= exp(deltan);
      windowUpdates_ = 0;
      windowAcceptance_ = 0;
    }

    // Proposal via log random walk
    param_ *= exp(random_.gaussian(0, tuning_));

    // Perform the non-centering
    float infecPartDiff = logLikelihood_.NonCentreInfecTimes(oldValue, param_, ncProp_);

    // Calculate candidate posterior
    float logPiCan = logLikelihood_.Propose() + log(param_.prior());

    // q-ratio
    float qratio = logf(param_ / oldValue);


    // Accept or reject
    if (log(random_.uniform()) < logPiCan - logPiCur + infecPartDiff + qratio)
      {
        acceptance_++;
        windowAcceptance_++;
        logLikelihood_.Accept();
      }
    else
      {
        param_ = oldValue;
        logLikelihood_.Reject();
      }

    numUpdates_++;
    windowUpdates_++;
  }



  InfectionTimeUpdate::InfectionTimeUpdate(const std::string& tag, Parameter& a, Parameter& b, const size_t reps, Random& random, McmcLikelihood& logLikelihood)
    : a_(a), b_(b), reps_(reps), McmcUpdate(tag, random, logLikelihood)
  {
    calls_.resize(3);
    accept_.resize(3);
    std::fill(calls_.begin(), calls_.end(), 0.0f);
    std::fill(accept_.begin(), accept_.end(), 0.0f);
  }

  InfectionTimeUpdate::~InfectionTimeUpdate()
  {

  }

  void
  InfectionTimeUpdate::Update()
  {
    for (size_t infec = 0; infec < reps_; ++infec)
      {
        float pickMove = random_.uniform();

        if(pickMove < 0.667) {
          accept_[0] += UpdateI();
	  calls_[0]++;
	}
        else if (pickMove < 0.833) {
          accept_[1] += AddI();
	  calls_[1]++;
	}
        else {
          accept_[2] += DeleteI();
	  calls_[2]++;
	}
      }
  }

  bool
  InfectionTimeUpdate::UpdateI()
  {
    size_t index = random_.integer(logLikelihood_.GetNumInfecs());
    //double newIN = random_.gamma(INFECPROP_A, INFECPROP_B); // Independence sampler
    double oldIN = logLikelihood_.GetIN(index);
    double newIN = oldIN * exp(random_.gaussian(0.0f,TUNEIN));

    float piCur = logLikelihood_.GetCurrentValue();
    float piCan = logLikelihood_.UpdateI(index, newIN);

    if (index < logLikelihood_.GetNumKnownInfecs())
      { // Known infection
        piCan += log(gammapdf(newIN, a_, b_));
        piCur += log(gammapdf(oldIN, a_, b_));
      }
    else
      { // Occult
        piCan += log(1 - gammacdf(newIN, a_, b_));
        piCur += log(1 - gammacdf(oldIN, a_, b_));
      }

    double qRatio = log(newIN / oldIN);

    //log(gammapdf(oldIN, INFECPROP_A, INFECPROP_B) / gammapdf(newIN, INFECPROP_A, INFECPROP_B));

    double accept = piCan - piCur + qRatio;

    if (log(random_.uniform()) < accept)
      {
  #ifndef NDEBUG
        cerr << "ACCEPT" << endl;
  #endif
        // Update the infection
        logLikelihood_.Accept();
        return true;
      }
    else
      {
  #ifndef NDEBUG
        cerr << "REJECT" << endl;
  #endif
        logLikelihood_.Reject();
        return false;
      }
  }

  bool
  InfectionTimeUpdate::AddI()
  {
    size_t numSusceptible = logLikelihood_.GetNumPossibleOccults();

    if (numSusceptible == 0)
      return false;

    size_t index = random_.integer(numSusceptible);

    double inProp = random_.gamma(INFECPROP_A,INFECPROP_B);

    double logPiCur = logLikelihood_.GetCurrentValue();

    double logPiCan = logLikelihood_.AddI(index, inProp)
        + log(1.0 - gammacdf(inProp, a_, b_));

    double qRatio = log(
        (1.0 / (logLikelihood_.GetNumOccults() + 1))
            / ((1.0 / numSusceptible)
                * gammapdf(inProp, INFECPROP_A, INFECPROP_B)));

    double accept = logPiCan - logPiCur + qRatio;

    // Perform accept/reject step.
    if (log(random_.uniform()) < accept)
      {
  #ifndef NDEBUG
        cerr << "ACCEPT" << endl;
  #endif
        logLikelihood_.Accept();
        return true;
      }
    else
      {
  #ifndef NDEBUG
        cerr << "REJECT" << endl;
  #endif
        logLikelihood_.Reject();
        return false;
      }
  }

  bool
  InfectionTimeUpdate::DeleteI()
  {
    if (logLikelihood_.GetNumOccults() == 0)
      {
  #ifndef NDEBUG
        cerr << __FUNCTION__ << endl;
        cerr << "Occults empty. Not deleting" << endl;
  #endif
        return false;
      }

    size_t numSusceptible = logLikelihood_.GetNumPossibleOccults();

    size_t toRemove = random_.integer(logLikelihood_.GetNumOccults());

    float inTime = logLikelihood_.GetIN(logLikelihood_.GetNumKnownInfecs() + toRemove);
    float logPiCur = logLikelihood_.GetCurrentValue()
        + log(1 - gammacdf(inTime, a_, b_));

    float logPiCan = logLikelihood_.DeleteI(toRemove);
    double qRatio = log(
        (1.0 / (numSusceptible + 1)
            * gammapdf(inTime, INFECPROP_A, INFECPROP_B))
            / (1.0 / logLikelihood_.GetNumOccults()));

    // Perform accept/reject step.
    double accept = logPiCan - logPiCur + qRatio;

    if (log(random_.uniform()) < accept)
      {
  #ifndef NDEBUG
        cerr << "ACCEPT" << endl;
  #endif
        logLikelihood_.Accept();
        return true;
      }
    else
      {
  #ifndef NDEBUG
        cerr << "REJECT" << endl;
  #endif
        logLikelihood_.Reject();
        return false;
      }
  }

  std::map<std::string, float>
  InfectionTimeUpdate::GetAcceptance() const
  {
    std::map<std::string, float> rv;

    float* accept = new float[3];
    for(size_t i=0; i<calls_.size(); ++i)
        accept[i] = accept_[i] / calls_[i];

    rv.insert(make_pair(tag_+":moveInfec", accept[0]));
    rv.insert(make_pair(tag_+":addInfec", accept[1]));
    rv.insert(make_pair(tag_+":delInfec", accept[2]));

    return rv;
  }

  void
  InfectionTimeUpdate::ResetAcceptance()
  {
    std::fill(accept_.begin(), accept_.end(), 0.0f);
    std::fill(calls_.begin(), accept_.end(), 0.0f);
  }




  SellkeSerializer::SellkeSerializer(const std::string filename, Random& random, McmcLikelihood& logLikelihood)
    : McmcUpdate(filename, random, logLikelihood)
  {

    outfile_.open(filename.c_str(),ios::out);
    if(!outfile_.is_open()) {
        string msg = "Cannot open SellkeSerializer output file '";
        msg += filename;
        msg += "'";
        throw output_exception(msg.c_str());
    }
  }
  SellkeSerializer::~SellkeSerializer()
  {
    outfile_.close();
  }
  void
  SellkeSerializer::Update()
  {
//    std::vector< map<string, double> > pressures;
//    //logLikelihood_.integPressure=pressures;
//    //boost::mpi::gather(env_->comm_,logLikelihood_.integPressure, pressures, 0);
//
//    bool isFirst = true;
//    for(std::vector< map<string, double> >::const_iterator jt = pressures.begin();
//        jt != pressures.end();
//        jt++) {
//        for(map<string,double>::const_iterator it = jt->begin();
//            it != jt->end();
//            it++)
//          {
//            if(!isFirst) outfile_ << " ";
//            else isFirst = false;
//
//            if(env_->pop_.getById(it->first).getI() != POSINF)
//              outfile_ << it->first << ":0:" << it->second;
//            else outfile_ << it->first << ":1:" << it->second;
//          }
//        outfile_ << "\n";
//    }
  }


}
