/*************************************************************************
 *  ./src/mcmc/MCMCUpdater.hpp
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
 * MCMCUpdater.hpp
 *
 *  Created on: 25 Jan 2011
 *      Author: stsiab
 */

#ifndef MCMCUPDATER_HPP_
#define MCMCUPDATER_HPP_

#include <string>
#include "StochasticNode.hpp"
#include "Random.hpp"
#include "EmpCovar.hpp"
#include "Mcmc.hpp"

namespace EpiRisk
{

  // FWD DECLS
  class Likelihood;
  class Mcmc;

  // FUNCTORS
  struct ExpTransform
  {
    double
    operator()(const double x)
    {
      return exp(x);
    }
  };

  struct LogTransform
  {
    double
    operator()(const double x)
    {
      return log(x);
    }
  };

  // MCMC UPDATERS
  class McmcUpdate
  {
  public:
    McmcUpdate(const std::string& tag, Random& rng, Likelihood& logLikelihood,
        Mcmc* const env);
    virtual
    ~McmcUpdate();
    virtual
    void
    update() = 0;

    //! Returns the MH acceptance probability
    virtual
    double
    getAcceptance() const;
    virtual std::string
    getTag() const;

  protected:
    const std::string tag_;
    Mcmc* env_;
    Likelihood& logLikelihood_;
    Random& random_;
    size_t acceptance_;
    size_t numUpdates_;
  };

  class SingleSiteLogMRW : public McmcUpdate
  {
    Parameter& param_;
    const double tuning_;
  public:
    SingleSiteLogMRW(const std::string& tag, Parameter& param,
        const double tuning, Random& rng, Likelihood& logLikelihood,
        Mcmc* const env);
    ~SingleSiteLogMRW();
    void
    update();
  };

  //! Adaptive Multisite Linear Random Walk algorithm
  class AdaptiveMultiMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<Identity>::CovMatrix Covariance;
    AdaptiveMultiMRW(const std::string& tag, UpdateBlock& params,
        size_t burnin, Random& rng, Likelihood& logLikelihood, Mcmc* const env);
    ~AdaptiveMultiMRW();
    void
    setCovariance(EmpCovar<Identity>::CovMatrix& covariance);
    Covariance
    getCovariance() const;
    void
    update();
  private:
    UpdateBlock& updateGroup_;
    size_t burnin_;
    EmpCovar<Identity>* empCovar_;
    EmpCovar<Identity>::CovMatrix* stdCov_;
  };

  //! Adaptive Multisite Logarithmic Random Walk algorithm
  class AdaptiveMultiLogMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    AdaptiveMultiLogMRW(const std::string& tag, UpdateBlock& params,
        size_t burnin, Random& rng, Likelihood& logLikelihood, Mcmc* const env);
    ~AdaptiveMultiLogMRW();
    void
    setCovariance(EmpCovar<LogTransform>::CovMatrix& covariance);
    Covariance
    getCovariance() const;
    void
    update();
  private:
    UpdateBlock& updateGroup_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;
  };

  //! Species MRW is a non-centred Multisite update for species inf/susc
  class SpeciesMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    SpeciesMRW(const std::string& tag, UpdateBlock& params,
        std::vector<double>& constants, size_t burnin, Random& rng, Likelihood& logLikelihood,
        Mcmc* const env);
    ~SpeciesMRW();
    void
    update();
    Covariance
    getCovariance() const;
  private:
    UpdateBlock& updateGroup_;
    UpdateBlock transformedGroup_;
    std::vector<double> constants_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;

  };

  //! InfectivityMRW is a non-centred Multisite update for species infectivity
  class InfectivityMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    InfectivityMRW(const std::string& tag, UpdateBlock& params,
        UpdateBlock& powers, size_t burnin, Random& rng, Likelihood& logLikelihood,
        Mcmc* const env);
    ~InfectivityMRW();
    void
    update();
    Covariance
    getCovariance() const;
  private:
    UpdateBlock& updateGroup_;
    UpdateBlock transformedGroup_;
    UpdateBlock powers_;
    std::vector<double> constants_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;

  };


  //! SusceptibilityMRW is a non-centred Multisite update for species inf/susc
  class SusceptibilityMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    SusceptibilityMRW(const std::string& tag, UpdateBlock& params,
        UpdateBlock& powers, size_t burnin, Random& rng, Likelihood& logLikelihood,
        Mcmc* const env);
    ~SusceptibilityMRW();
    void
    update();
    Covariance
    getCovariance() const;
  private:
    UpdateBlock& updateGroup_;
    UpdateBlock transformedGroup_;
    UpdateBlock powers_;
    std::vector<double> constants_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;

  };


  class InfectivityPowMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    InfectivityPowMRW(const std::string& tag, UpdateBlock& params,
        size_t burnin, Random& rng, Likelihood& logLikelihood, Mcmc* const env);
    ~InfectivityPowMRW();
    void
    setCovariance(EmpCovar<LogTransform>::CovMatrix& covariance);
    Covariance
    getCovariance() const;
    void
    update();
  private:
    UpdateBlock& updateGroup_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;
    ublas::matrix<double>* cache_;
  };


  class SusceptibilityPowMRW : public McmcUpdate
  {
  public:
    typedef EmpCovar<LogTransform>::CovMatrix Covariance;
    SusceptibilityPowMRW(const std::string& tag, UpdateBlock& params,
        size_t burnin, Random& rng, Likelihood& logLikelihood, Mcmc* const env);
    ~SusceptibilityPowMRW();
    void
    setCovariance(EmpCovar<LogTransform>::CovMatrix& covariance);
    Covariance
    getCovariance() const;
    void
    update();
  private:
    UpdateBlock& updateGroup_;
    size_t burnin_;
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;
    ublas::matrix<double>* cache_;
  };

  //! SellkeSerializer writes out integrated infectious pressure for each individual
  class SellkeSerializer : public McmcUpdate
  {
  public:
    SellkeSerializer(const std::string, Random& rng, Likelihood& logLikelihood, Mcmc* const env);
    virtual
    ~SellkeSerializer();
    void
    update();
  private:
    ofstream outfile_;
  };

}
#endif /* MCMCUPDATER_HPP_ */
