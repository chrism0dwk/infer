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
    McmcUpdate(const std::string& tag, Random& rng,
        Likelihood& logLikelihood, Mcmc* const env );
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


  class SingleSiteMRW : public McmcUpdate
  {
    Parameter& param_;
    const double tuning_;
  public:
    SingleSiteMRW(const std::string& tag, Parameter& param, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* const env );
    ~SingleSiteMRW();
    void
    update();
  };


  class SingleSiteLogMRW : public McmcUpdate
  {
    Parameter& param_;
    const double tuning_;
  public:
    SingleSiteLogMRW(const std::string& tag, Parameter& param, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* const env );
    ~SingleSiteLogMRW();
    void
    update();
  };

  //! Adaptive Multisite Linear Random Walk algorithm
  class AdaptiveMultiMRW : public McmcUpdate
    {
    public:
      typedef EmpCovar<Identity>::CovMatrix Covariance;
      AdaptiveMultiMRW(const std::string& tag, UpdateBlock& params, size_t burnin, Random& rng,
          Likelihood& logLikelihood, Mcmc* const env  );
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
    AdaptiveMultiLogMRW(const std::string& tag, UpdateBlock& params, size_t burnin, Random& rng,
        Likelihood& logLikelihood, Mcmc* const env  );
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



  //! Single site update for within farm epidemic
  class WithinFarmBetaLogMRW : public McmcUpdate
  {
  public:
    WithinFarmBetaLogMRW(Parameter& param,  const double alpha, const double gamma, Population<TestCovars>& pop_, const double tuning, Random& rng, Likelihood& logLikelihood, Mcmc* const env );
    ~WithinFarmBetaLogMRW();
    void
    update();
  private:
    Parameter& param_;
    const double tuning_;
    const double alpha_;
    const double gamma_;
    Population<TestCovars>& pop_;
  };

}
#endif /* MCMCUPDATER_HPP_ */
