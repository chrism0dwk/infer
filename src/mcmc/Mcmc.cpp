/*************************************************************************
 *  ./src/mcmc/Mcmc.cpp
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

#include <cmath>
#include <boost/numeric/ublas/io.hpp>
#include <ctime>
#include <algorithm>
#include <functional>
#include <vector>
#include <iomanip>
#include <gsl/gsl_cdf.h>
#include <sys/time.h>

//#ifdef __LINUX__
//#include <acml_mv.h>
//#define log fastlog
//#define exp fastexp
//#endif

#include "Mcmc.hpp"

using namespace EpiRisk;

// Constants
const double tuneI = 2.5;

inline
double
timeinseconds(const timeval a, const timeval b)
{
  timeval result;
  timersub(&b, &a, &result);
  return result.tv_sec + result.tv_usec / 1000000.0;
}

inline
double
onlineMean(const double x, const double xbar, const double n)
{
  return xbar + (x - xbar) / n;
}



Mcmc::Mcmc(GpuLikelihood& likelihood, const size_t randomSeed) :
    likelihood_(likelihood), random_(NULL), timeCalc_(0.0), timeUpdate_(
        0.0)
{

  random_ = new Random(randomSeed);
}

Mcmc::~Mcmc()
{
  delete random_;
}

//! Pushes an updater onto the MCMC stack
SingleSiteLogMRW*
Mcmc::NewSingleSiteLogMRW(Parameter& param, const double tuning)
{
  SingleSiteLogMRW* update = new SingleSiteLogMRW(param.GetTag(), param, tuning,
      *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an updater onto the MCMC stack
AdaptiveMultiLogMRW*
Mcmc::NewAdaptiveMultiLogMRW(const string name, UpdateBlock& updateGroup,
    size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<LogTransform>::CovMatrix initCov(updateGroup.size());
  for (size_t i = 0; i < updateGroup.size(); ++i)
    for (size_t j = 0; j <= i; ++j)
      initCov(i, j) = i == j ? 0.1 : 0.0;

  AdaptiveMultiLogMRW* update = new AdaptiveMultiLogMRW(name, updateGroup,
      burnin, *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an updater onto the MCMC stack
AdaptiveMultiMRW*
Mcmc::NewAdaptiveMultiMRW(const string name, UpdateBlock& updateGroup,
    const size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<Identity>::CovMatrix initCov(updateGroup.size());
  for (size_t i = 0; i < updateGroup.size(); ++i)
    for (size_t j = 0; j <= i; ++j)
      initCov(i, j) = i == j ? 0.1 : 0.0;

  AdaptiveMultiMRW* update = new AdaptiveMultiMRW(name, updateGroup, burnin,
      *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an SpeciesMRW updater onto the MCMC stack
SpeciesMRW*
Mcmc::NewSpeciesMRW(const string tag, UpdateBlock& params,
    std::vector<double>& alpha, const size_t burnin)
{
  SpeciesMRW* update = new SpeciesMRW(tag, params, alpha, burnin, *random_,
      likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an SusceptibilityMRW updater onto the MCMC stack
SusceptibilityMRW*
Mcmc::NewSusceptibilityMRW(const string tag, UpdateBlock& params,
    const size_t burnin)
{
  SusceptibilityMRW* update = new SusceptibilityMRW(tag, params, burnin,
      *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an InfectivityMRW updater onto the MCMC stack
InfectivityMRW*
Mcmc::NewInfectivityMRW(const string tag, UpdateBlock& params,
    const size_t burnin)
{
  InfectivityMRW* update = new InfectivityMRW(tag, params, burnin,
      *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an infectious period scale updater onto the MCMC stack
InfectionTimeGammaScale*
Mcmc::NewInfectionTimeGammaScale(const string tag, Parameter& param, const float tuning)
{
  InfectionTimeGammaScale* update = new InfectionTimeGammaScale(tag, param, tuning, *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an infection time updater on the the MCMC stack
InfectionTimeUpdate*
Mcmc::NewInfectionTimeUpdate(const string tag, Parameter& a, Parameter& b, const size_t reps)
{
  InfectionTimeUpdate* update = new InfectionTimeUpdate(tag, a, b, reps, *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes a SellkeSerializer onto the MCMC stack
SellkeSerializer*
Mcmc::NewSellkeSerializer(const string filename)
{
  SellkeSerializer* update = new SellkeSerializer(filename, *random_,
      likelihood_);
  updateStack_.push_back(update);
  return update;
}

//! Performs a parameter sweep
void
Mcmc::Update()
{
  // Performs a sweep over the parameters
  //  this could be generalized!

  try
    {

      for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin();
          it != updateStack_.end(); ++it)
        {
          it->Update();
        }
    }
  catch (logic_error& e)
    {
      cerr << "Logic Error occurred: " << e.what() << endl;
      throw e;
    }
  catch (exception& e)
    {
      cerr << "Unknown error in " << __FILE__ << ":" << __LINE__ << ": "
          << e.what() << endl;
      throw e;
    }
}

map<string, float>
Mcmc::GetAcceptance() const
{
  map<string, float> acceptance;
  for (boost::ptr_list<McmcUpdate>::const_iterator it = updateStack_.begin();
      it != updateStack_.end(); ++it)
    {
      map<string, float> tmp = it->GetAcceptance();
      acceptance.insert(tmp.begin(), tmp.end());
    }

  return acceptance;
}

void
Mcmc::ResetAcceptance()
{
  for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin();
      it != updateStack_.end(); ++it)
    it->ResetAcceptance();
}

