/*
 * MCMCUpdater.hpp
 *
 *  Created on: 25 Jan 2011
 *      Author: stsiab
 */

#ifndef MCMCUPDATER_HPP_
#define MCMCUPDATER_HPP_

#include <string>

#include "Parameter.hpp"
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
    McmcUpdate(const std::string& tag, ParameterView& params, Random& rng,
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
    ParameterView updateGroup_;
    size_t acceptance_;
    size_t numUpdates_;
  };

  //! Adaptive Multisite Logarithmic Random Walk algorithm
  class AdaptiveMultiLogMRW : public McmcUpdate
  {
  public:
    AdaptiveMultiLogMRW(const std::string& tag, ParameterView& params, Random& rng,
        Likelihood& logLikelihood, Mcmc* const env  );
    ~AdaptiveMultiLogMRW();
    void
    setCovariance(EmpCovar<LogTransform>::CovMatrix& covariance);
    void
    update();
  private:
    EmpCovar<LogTransform>* empCovar_;
    EmpCovar<LogTransform>::CovMatrix* stdCov_;
  };

}
#endif /* MCMCUPDATER_HPP_ */
