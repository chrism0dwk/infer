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
/* ./src/mcmc/MCMCUpdater.cpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */
/*
 * MCMCUpdater.cpp
 *
 *  Created on: 21 Jan 2011
 *      Author: stsiab
 */

#include <algorithm>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <iomanip>

#include "MCMCUpdater.hpp"
#include "StochasticNode.hpp"

#define TUNEIN 2.5

#define INFECPROP_A 2.0
#define INFECPROP_B 0.15

namespace EpiRisk
{
  namespace Mcmc
  {

    bool
    _checkNotLessThanZero(const UpdateBlock& parms)
    {
      int lessThanZero = 0;
      for (size_t i = 0; i < parms.size(); ++i)
        if (parms[i].getValue() < 0.0)
          lessThanZero++;

      if (lessThanZero == 0)
        return true;
      else
        return false;
    }

    int
    _determinant_sign(const ublas::permutation_matrix<std::size_t>& pm)
    {
      int pm_sign = 1;
      std::size_t size = pm.size();
      for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
          pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
      return pm_sign;
    }

    template<class T>
      float
      _determinant(const T& mat)
      {
        ublas::matrix<float> m = mat;
        ublas::permutation_matrix<size_t> pm(m.size1());
        float det = 1.0;
        if (ublas::lu_factorize(m, pm))
          {
            det = 0.0;
          }
        else
          {
            for (int i = 0; i < m.size1(); i++)
              det *= m(i, i); // multiply by elements on diagonal
            det = det * _determinant_sign(pm);
          }
        return det;
      }

    inline
    float
    extremepdf(const float x, const float a, const float b)
    {
      return a * b * exp(a + b * x - a * exp(b * x));
    }

    inline
    float
    extremecdf(const float x, const float a, const float b)
    {
      return 1 - exp(-a * (exp(b * x) - 1));
    }

    inline
    float
    gammacdf(const float x, const float a, const float b)
    {
      return gsl_cdf_gamma_P(x, a, 1.0 / b);
    }

    inline
    float
    gammapdf(const float x, const float a, const float b)
    {
      return gsl_ran_gamma_pdf(x, a, 1.0 / b);
    }

    inline
    float
    gaussianTailPdf(const float x, const float mean, const float var)
    {
      return gsl_ran_gaussian_tail_pdf(x - mean, -mean, sqrt(var));
    }

    McmcUpdate::McmcUpdate() :
        acceptance_(0), numUpdates_(0), params_(NULL)
    {
    }

    McmcUpdate::~McmcUpdate()
    {
    }

    void
    McmcUpdate::SetParameters(UpdateBlock& parameters)
    {
      params_ = &parameters;
    }

    std::map<std::string, float>
    McmcUpdate::GetAcceptance() const
    {
      std::map<std::string, float> accepts;
      accepts.insert(
          std::make_pair(tag_, (float) acceptance_ / (float) numUpdates_));
      return accepts;
    }

    void
    McmcUpdate::ResetAcceptance()
    {
      acceptance_ = 0;
      numUpdates_ = 0;
    }

    SingleSiteLogMRW::SingleSiteLogMRW() :
        tuning_(0.1)
    {
    }

    void
    SingleSiteLogMRW::SetTuning(const float tuning)
    {
      tuning_ = tuning;
    }

    void
    SingleSiteLogMRW::Update()
    {
      Parameter& param_((*params_)[0].getParameter());

      float oldValue = param_;

      // Calculate current posterior
      float logPiCur = likelihood_->GetCurrentValue() + log(param_.prior());

      // Proposal via log random walk
      param_ *= exp(random_->gaussian(0, tuning_));

      // Calculate candidate posterior
      float logPiCan = likelihood_->Propose() + log(param_.prior());

      // q-ratio
      float qratio = param_ / oldValue;

      // Accept or reject
      if (log(random_->uniform()) < logPiCan - logPiCur + qratio)
        {
          likelihood_->Accept();
          acceptance_++;
        }
      else
        {
          param_ = oldValue;
          likelihood_->Reject();
        }

      numUpdates_++;

    }

    void
    AdaptiveMultiMRW::Update()
    {
      // Save old values
      std::vector<float> oldParams(params_->size());
      for (size_t i = 0; i < params_->size(); i++)
        oldParams[i] = (*params_)[i].getValue();

      // Update empirical covariance
      empCovar_->sample();

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.234)
            adaptScalar_ *= exp(-deltan);
          else
            adaptScalar_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Calculate current posterior
      float logPiCur = likelihood_->GetCurrentValue();
      for (size_t p = 0; p < params_->size(); ++p)
        logPiCur += log((*params_)[p].prior());

      // Propose as in Haario, Sachs, Tamminen (2001)
      Random::Variates vars;
      if (random_->uniform() < 0.95 and numUpdates_ > burnin_)
        {
          try
            {
              vars = random_->mvgauss(
                  empCovar_->getCovariance() * adaptScalar_ / params_->size());
            }
          catch (cholesky_error& e)
            {
              vars = random_->mvgauss(*stdCov_);
            }
        }
      else
        vars = random_->mvgauss(*stdCov_);

      // Log MRW proposal
      for (size_t p = 0; p < params_->size(); ++p)
        (*params_)[p].setValue((*params_)[p].getValue() + vars[p]);

      // Calculate candidate posterior
      float logPiCan = likelihood_->Propose();
      for (size_t p = 0; p < params_->size(); ++p)
        logPiCan += log((*params_)[p].prior());

      // Proposal ratio
      float qRatio = 0.0; // Gaussian proposal cancels

      // Accept or reject
      float accept = logPiCan - logPiCur + qRatio;
      if (log(random_->uniform()) < accept)
        {
          likelihood_->Accept();
          acceptance_++;
          windowAcceptance_++;
        }
      else
        {
          likelihood_->Reject();
          for (size_t p = 0; p < params_->size(); ++p)
            (*params_)[p].setValue(oldParams[p]);
        }

      ++numUpdates_;
      ++windowUpdates_;
    }

    void
    AdaptiveMultiLogMRW::Update()
    {
      // Save old values
      std::vector<float> oldParams(params_->size());
      for (size_t i = 0; i < params_->size(); i++)
        oldParams[i] = (*params_)[i].getValue();

      // Update empirical covariance
      empCovar_->sample();

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.234)
            adaptScalar_ *= exp(-deltan);
          else
            adaptScalar_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Calculate current posterior
      float logPiCur = likelihood_->GetCurrentValue();
      for (size_t p = 0; p < params_->size(); ++p)
        logPiCur += log((*params_)[p].prior());

      // Propose as in Haario, Sachs, Tamminen (2001)
      Random::Variates logvars;
      if (random_->uniform() < 0.95 and numUpdates_ > burnin_)
        {
          try
            {
              logvars = random_->mvgauss(
                  empCovar_->getCovariance() * adaptScalar_ / params_->size());
            }
          catch (cholesky_error& e)
            {
              logvars = random_->mvgauss(*stdCov_);
            }
        }
      else
        logvars = random_->mvgauss(*stdCov_);

      // Log MRW proposal
      for (size_t p = 0; p < params_->size(); ++p)
        (*params_)[p].setValue((*params_)[p].getValue() * exp(logvars[p]));

      // Calculate candidate posterior

      float logPiCan = likelihood_->Propose();
      for (size_t p = 0; p < params_->size(); ++p)
        logPiCan += log((*params_)[p].prior());

      // Proposal ratio
      float qRatio = 0.0;
      for (size_t p = 0; p < params_->size(); ++p)
        qRatio += log((*params_)[p].getValue() / oldParams[p]);

      // Accept or reject
      float accept = logPiCan - logPiCur + qRatio;
      if (log(random_->uniform()) < accept)
        {
          likelihood_->Accept();
          acceptance_++;
          windowAcceptance_++;
        }
      else
        {
          likelihood_->Reject();
          for (size_t p = 0; p < params_->size(); ++p)
            (*params_)[p].setValue(oldParams[p]);
        }

      ++numUpdates_;
      ++windowUpdates_;
    }

    void
    InfectivityMRW::SetParameters(UpdateBlock& params)
    {

      params_ = &params;
      constants_.resize(3, 0.0);
      transformedGroup_.add(params[1]);
      transformedGroup_.add(params[2]);

      InitCovariance(transformedGroup_);

    }

    void
    InfectivityMRW::Update()
    {
      // Save parameters
      std::vector<float> oldParams(params_->size());
      for (size_t i = 0; i < params_->size(); ++i)
        oldParams[i] = (*params_)[i].getValue();

      // Calculate constants
      likelihood_->GetSumInfectivityPow(&constants_[0]);

      // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
      float R = (*params_)[0].getValue()
          * (constants_[0] + (*params_)[1].getValue() * constants_[1]
              + (*params_)[2].getValue() * constants_[2]);

      // Current posterior
      float logPiCur = likelihood_->GetCurrentValue()
          + log((*params_)[0].prior()) + log((*params_)[1].prior())
          + log((*params_)[2].prior());

      // Make proposal
      ublas::vector<float> transform(params_->size());
      transform(0) = (*params_)[0].getValue() * constants_[0];
      transform(1) = (*params_)[0].getValue() * (*params_)[1].getValue()
          * constants_[1];
      transform(2) = (*params_)[0].getValue() * (*params_)[2].getValue()
          * constants_[2];

      // Sample transformed posterior
      ublas::vector<float> sample = ublas::vector_range<ublas::vector<float> >(
          transform, ublas::range(1, transform.size()));
      empCovar_->sample(sample);

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.234)
            adaptScalar_ *= exp(-deltan);
          else
            adaptScalar_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Propose as in Haario, Sachs, Tamminen (2001)
      Random::Variates logvars;
      if (random_->uniform() < 0.95 and numUpdates_ > burnin_)
        {
          try
            {
              Covariance tmp = empCovar_->getCovariance();
              //tmp = tmp /  _determinant(tmp);
              logvars = random_->mvgauss(
                  tmp * adaptScalar_ / transformedGroup_.size());
            }
          catch (cholesky_error& e)
            {
              cerr << "Cholesky error in " << __PRETTY_FUNCTION__ << ": '"
                  << e.what() << "'" << endl;
              logvars = random_->mvgauss(*stdCov_);
            }
        }
      else
        logvars = random_->mvgauss(*stdCov_);

      transform(1) *= exp(logvars(0));
      transform(2) *= exp(logvars(1));
      transform(0) = R - transform(1) - transform(2);

      // Reject if we get a neg value
      if (transform(0) < 0.0f)
        {
          for (size_t i = 0; i < params_->size(); ++i)
            (*params_)[i].setValue(oldParams[i]);
          ++numUpdates_;
          ++windowUpdates_;
          return;
        }

      // Transform back
      (*params_)[0].setValue(transform(0) / constants_[0]);
      (*params_)[1].setValue(
          transform(1) / ((*params_)[0].getValue() * constants_[1]));
      (*params_)[2].setValue(
          transform(2) / ((*params_)[0].getValue() * constants_[2]));

      // Calculate candidate posterior
      float logPiCan = likelihood_->Propose() + log((*params_)[0].prior())
          + log((*params_)[1].prior()) + log((*params_)[2].prior());

      // q-Ratio
      float qRatio = log(
          transform(1) / (oldParams[0] * oldParams[1] * constants_[1]))
          + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

      // Accept/reject
      if (log(random_->uniform()) < logPiCan - logPiCur + qRatio)
        {
          likelihood_->Accept();
          acceptance_++;
          windowAcceptance_++;
        }
      else
        {
          for (size_t i = 0; i < params_->size(); ++i)
            (*params_)[i].setValue(oldParams[i]);
          likelihood_->Reject();
        }

      ++numUpdates_;
      ++windowUpdates_;

    }

    void
    SusceptibilityMRW::SetParameters(UpdateBlock& params)
    {

      params_ = &params;
      constants_.resize(3, 0.0);
      transformedGroup_.add(params[1]);
      transformedGroup_.add(params[2]);

      InitCovariance(transformedGroup_);
    }

    void
    SusceptibilityMRW::Update()
    {
      // Save parameters
      std::vector<float> oldParams(params_->size());
      for (size_t i = 0; i < params_->size(); ++i)
        oldParams[i] = (*params_)[i].getValue();

      // Calculate constants
      likelihood_->GetSumSusceptibilityPow(&constants_[0]);

      // Calculate sum of infectious pressure: gamma*(cattle + xi_s*sheep + xi_p*pigs)
      float R = (*params_)[0].getValue()
          * (constants_[0] + (*params_)[1].getValue() * constants_[1]
              + (*params_)[2].getValue() * constants_[2]);

      // Current posterior
      float logPiCur = likelihood_->GetCurrentValue()
          + log((*params_)[0].prior()) + log((*params_)[1].prior())
          + log((*params_)[2].prior());

      // Make proposal
      ublas::vector<float> transform(params_->size());
      transform(0) = (*params_)[0].getValue() * constants_[0];
      transform(1) = (*params_)[0].getValue() * (*params_)[1].getValue()
          * constants_[1];
      transform(2) = (*params_)[0].getValue() * (*params_)[2].getValue()
          * constants_[2];

      // Sample transformed posterior
      ublas::vector<float> sample = ublas::vector_range<ublas::vector<float> >(
          transform, ublas::range(1, transform.size()));
      empCovar_->sample(sample);

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.234)
            adaptScalar_ *= exp(-deltan);
          else
            adaptScalar_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Propose as in Haario, Sachs, Tamminen (2001)
      Random::Variates logvars;
      if (random_->uniform() < 0.95 and numUpdates_ > burnin_)
        {
          try
            {
              logvars = random_->mvgauss(
                  empCovar_->getCovariance() * adaptScalar_
                      / transformedGroup_.size());
            }
          catch (cholesky_error& e)
            {
              logvars = random_->mvgauss(*stdCov_);
            }
        }
      else
        logvars = random_->mvgauss(*stdCov_);

      // Use indep gaussians here
      transform(1) *= exp(logvars(0)); //exp(random_.gaussian(0,0.8));
      transform(2) *= exp(logvars(1)); //exp(random_.gaussian(0,0.1));
      transform(0) = R - transform(1) - transform(2);

      // Reject if we get a neg value
      if (transform(0) < 0.0f)
        {
          for (size_t i = 0; i < params_->size(); ++i)
            (*params_)[i].setValue(oldParams[i]);
          ++numUpdates_;
          ++windowUpdates_;
          return;
        }

      // Transform back
      (*params_)[0].setValue(transform(0) / constants_[0]);
      (*params_)[1].setValue(
          transform(1) / ((*params_)[0].getValue() * constants_[1]));
      (*params_)[2].setValue(
          transform(2) / ((*params_)[0].getValue() * constants_[2]));

      // Calculate candidate posterior
      float logPiCan = likelihood_->Propose() + log((*params_)[0].prior())
          + log((*params_)[1].prior()) + log((*params_)[2].prior());

      // q-Ratio
      float qRatio = log(
          transform(1) / (oldParams[0] * oldParams[1] * constants_[1]))
          + log(transform(2) / (oldParams[0] * oldParams[2] * constants_[2]));

      // Accept/reject
      if (log(random_->uniform()) < logPiCan - logPiCur + qRatio)
        {
          likelihood_->Accept();
          acceptance_++;
          ++windowAcceptance_;
        }
      else
        {
          for (size_t i = 0; i < params_->size(); ++i)
            (*params_)[i].setValue(oldParams[i]);
          likelihood_->Reject();
        }

      ++numUpdates_;
      ++windowUpdates_;

    }

    InfectionTimeGammaCentred::InfectionTimeGammaCentred() :
        tuning_(ADAPTIVESCALE), windowUpdates_(0), windowAcceptance_(0)
    {

    }

    void
    InfectionTimeGammaCentred::SetTuning(const float tuning)
    {
      tuning_ = tuning;
    }

    void
    InfectionTimeGammaCentred::Update()
    {

      Parameter& param_((*params_)[0].getParameter());

      float oldValue = param_;

      // Calculate current posterior
      float logPiCur = likelihood_->GetInfectionPart() + log(param_.prior());

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.44)
            tuning_ *= exp(-deltan);
          else
            tuning_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Proposal via log random walk
      param_ *= exp(random_->gaussian(0, tuning_));

      // Calculate candidate posterior
      float logPiCan = likelihood_->GetInfectionPart() + log(param_.prior());

      // q-ratio
      float qratio = logf(param_ / oldValue);

      // Accept or reject
      if (log(random_->uniform()) < logPiCan - logPiCur + qratio)
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

    InfectionTimeGammaNC::InfectionTimeGammaNC() :
        ncProp_(0.3), tuning_(ADAPTIVESCALE), windowUpdates_(0), windowAcceptance_(
            0)
    {

    }

    InfectionTimeGammaNC::~InfectionTimeGammaNC()
    {

    }

    void
    InfectionTimeGammaNC::SetNCRatio(const float ncProp)
    {
      ncProp_ = ncProp;
    }

    void
    InfectionTimeGammaNC::SetTuning(const float tuning)
    {
      tuning_ = tuning;
    }

    void
    InfectionTimeGammaNC::Update()
    {

      Parameter& param_((*params_)[0].getParameter());

      float oldValue = param_;

      // Calculate current posterior
      float logPiCur = likelihood_->GetCurrentValue() + log(param_.prior());

      // Adapt adaptscalar
      if (windowUpdates_ % WINDOWSIZE == 0)
        {
          float accept = (float) windowAcceptance_ / (float) windowUpdates_;
          float deltan = min(0.5, 1.0 / sqrtf(numUpdates_ / WINDOWSIZE));
          if (accept < 0.44)
            tuning_ *= exp(-deltan);
          else
            tuning_ *= exp(deltan);
          windowUpdates_ = 0;
          windowAcceptance_ = 0;
        }

      // Proposal via log random walk
      param_ *= exp(random_->gaussian(0, tuning_));

      // Perform the non-centering
      float infecPartDiff = likelihood_->NonCentreInfecTimes(oldValue, param_,
          ncProp_);

      // Calculate candidate posterior
      float logPiCan = likelihood_->Propose() + log(param_.prior());

      // q-ratio
      float qratio = logf(param_ / oldValue);

      // Accept or reject
      if (log(random_->uniform())
          < logPiCan - logPiCur + infecPartDiff + qratio)
        {
          acceptance_++;
          windowAcceptance_++;
          likelihood_->Accept();
        }
      else
        {
          param_ = oldValue;
          likelihood_->Reject();
        }

      numUpdates_++;
      windowUpdates_++;
    }

    InfectionTimeUpdate::InfectionTimeUpdate() :
        reps_(1), ucalls_(0), doCompareProductVector_(NULL), updateTuning_(TUNEIN)
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
    InfectionTimeUpdate::SetReps(const size_t reps)
    {
      reps_ = reps;
    }

    void
    InfectionTimeUpdate::Update()
    {
      for (size_t infec = 0; infec < reps_; ++infec)
        {
          float pickMove = random_->uniform(0.0f, 1.0f);

          if (pickMove < 0.05f)
            {
              accept_[0] += UpdateI();
              calls_[0]++;
            }
          else if (pickMove < 0.525f)
            {
              accept_[1] += AddI();
              calls_[1]++;
            }
          else
            {
              accept_[2] += DeleteI();
              calls_[2]++;
            }

          if (*doCompareProductVector_)
            {
              float proposal = likelihood_->Propose();

              float likdiff = fabs(
                  (proposal - likelihood_->GetCurrentValue())
                      / likelihood_->GetCurrentValue());

              if (true)
                {
                  std::stringstream s;
                  s << "Likelihood discrepancy! Updated: "
                      << likelihood_->GetCurrentValue() << "; recalc: "
                      << proposal << " (" << likdiff << ")";

                  likelihood_->CompareProdVectors();

                  const GpuLikelihood::LikelihoodComponents* pLik =
                      likelihood_->GetProposal();
                  const GpuLikelihood::LikelihoodComponents* cLik =
                      likelihood_->GetCurrent();
                  cerr << setprecision(6);
                  cerr << "bgIntegral: " << pLik->bgIntegral << "\t"
                      << cLik->bgIntegral << endl;
                  cerr << "integral: " << pLik->integral << "\t"
                      << cLik->integral << endl;
                  cerr << "product: " << pLik->logProduct << "\t"
                      << cLik->logProduct << endl;
                  cerr << "Likelihood: " << proposal << "\t" << likelihood_->GetCurrentValue();
                  likelihood_->Reject();
                  //throw logic_error(s.str().c_str());
                }
              *doCompareProductVector_ = false;
            }
        }
      ucalls_++;
    }

    bool
    InfectionTimeUpdate::UpdateI()
    {

#ifndef NDEBUG
      std::cerr << "UPDATE" << std::endl;
#endif

      Parameter& a_((*params_)[0].getParameter());
      Parameter& b_((*params_)[1].getParameter());

      size_t index = random_->integer(likelihood_->GetNumInfecs());
      //float newIN = random_->gamma(INFECPROP_A, INFECPROP_B); // Independence sampler
      float oldIN = likelihood_->GetIN(index);
      float newIN = oldIN * exp(random_->gaussian(0.0f, updateTuning_));

      float piCur = likelihood_->GetCurrentValue();
      float piCan = likelihood_->UpdateI(index, newIN);

      if (index < likelihood_->GetNumKnownInfecs())
        { // Known infection
          piCan += log(gammapdf(newIN, a_, b_));
          piCur += log(gammapdf(oldIN, a_, b_));
        }
      else
        { // Occult
          piCan += log(1 - gammacdf(newIN, a_, b_));
          piCur += log(1 - gammacdf(oldIN, a_, b_));
        }

      float qRatio = log(newIN / oldIN);

      //float qRatio = log(gammapdf(oldIN, INFECPROP_A, INFECPROP_B) / gammapdf(newIN, INFECPROP_A, INFECPROP_B));

      float accept = piCan - piCur + qRatio;

      if (log(random_->uniform()) < accept)
        {
#ifndef NDEBUG
          cerr << "ACCEPT" << endl;
#endif
          // Update the infection
          likelihood_->Accept();
          return true;
        }
      else
        {
#ifndef NDEBUG
          cerr << "REJECT" << endl;
#endif
          likelihood_->Reject();
          return false;
        }
    }

    bool
    InfectionTimeUpdate::AddI()
    {

#ifndef NDEBUG
      std::cerr << "ADD" << std::endl;
#endif

      Parameter& a_((*params_)[0].getParameter());
      Parameter& b_((*params_)[1].getParameter());

      size_t numSusceptible = likelihood_->GetNumPossibleOccults();

      if (numSusceptible == 0)
        return false;

      size_t index = random_->integer(numSusceptible);

      float inProp = random_->gamma(INFECPROP_A, INFECPROP_B);

      float logPiCur = likelihood_->GetCurrentValue();

      float logLikCan = likelihood_->AddI(index, inProp);

      float logPiCan = logLikCan + log(1.0 - gammacdf(inProp, a_, b_));

      float qRatio = log(
          (1.0 / (likelihood_->GetNumOccults() + 1))
              / ((1.0 / numSusceptible)
                  * gammapdf(inProp, INFECPROP_A, INFECPROP_B)));

      float accept = logPiCan - logPiCur + qRatio;

      // Perform accept/reject step.
      if (log(random_->uniform()) < accept)
        {
#ifndef NDEBUG
          cerr << "ACCEPT" << endl;
#endif
          likelihood_->Accept();
          return true;
        }
      else
        {
#ifndef NDEBUG
          cerr << "REJECT" << endl;
#endif
          likelihood_->Reject();
          return false;
        }
    }

    bool
    InfectionTimeUpdate::DeleteI()
    {

#ifndef NDEBUG
      std::cerr << "DEL" << std::endl;
#endif

      Parameter& a_((*params_)[0].getParameter());
      Parameter& b_((*params_)[1].getParameter());

      if (likelihood_->GetNumOccults() == 0)
        {
#ifndef NDEBUG
          cerr << __FUNCTION__ << endl;
          cerr << "Occults empty. Not deleting" << endl;
#endif
          return false;
        }

      size_t numSusceptible = likelihood_->GetNumPossibleOccults();

      size_t toRemove = random_->integer(likelihood_->GetNumOccults());

      float inTime = likelihood_->GetIN(
          likelihood_->GetNumKnownInfecs() + toRemove);
      float logPiCur = likelihood_->GetCurrentValue()
          + log(1 - gammacdf(inTime, a_, b_));

      float logPiCan = likelihood_->DeleteI(toRemove);
      float qRatio = log(
          (1.0 / (numSusceptible + 1)
              * gammapdf(inTime, INFECPROP_A, INFECPROP_B))
              / (1.0 / likelihood_->GetNumOccults()));

      // Perform accept/reject step.
      float accept = logPiCan - logPiCur + qRatio;

      if (log(random_->uniform()) < accept)
        {
#ifndef NDEBUG
          cerr << "ACCEPT" << endl;
#endif
          likelihood_->Accept();
          return true;
        }
      else
        {
#ifndef NDEBUG
          cerr << "REJECT" << endl;
#endif
          likelihood_->Reject();
          return false;
        }
    }

    std::map<std::string, float>
    InfectionTimeUpdate::GetAcceptance() const
    {
      std::map<std::string, float> rv;

      float* accept = new float[3];
      for (size_t i = 0; i < calls_.size(); ++i)
        accept[i] = accept_[i] / calls_[i];

      rv.insert(make_pair(tag_ + ":moveInfec", accept[0]));
      rv.insert(make_pair(tag_ + ":addInfec", accept[1]));
      rv.insert(make_pair(tag_ + ":delInfec", accept[2]));

      return rv;
    }

    void
    InfectionTimeUpdate::ResetAcceptance()
    {
      std::fill(accept_.begin(), accept_.end(), 0.0f);
      std::fill(calls_.begin(), accept_.end(), 0.0f);
    }

  }
}
