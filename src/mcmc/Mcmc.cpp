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
const double a = 0.05;//0.015;
const double b = 0.2;//0.8;
const double tuneI = 2.0;

inline
double
timeinseconds(const timeval a, const timeval b)
{
  timeval result;
  timersub(&b,&a,&result);
  return result.tv_sec + result.tv_usec / 1000000.0;
}

inline
double
onlineMean(const double x, const double xbar, const double n)
{
  return xbar + (x - xbar) / n;
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
gaussianTailPdf(const double x, const double mean, const double var)
{
  return gsl_ran_gaussian_tail_pdf(x - mean, -mean, sqrt(var));
}



Mcmc::Mcmc(GpuLikelihood& likelihood, const size_t randomSeed) :
  likelihood_(likelihood), random_(NULL), numIUpdates_(0), timeCalc_(0.0), timeUpdate_(0.0),
      numCalc_(0), numUpdate_(0)
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
  SingleSiteLogMRW* update = new SingleSiteLogMRW(param.getTag(), param,
      tuning, *random_, likelihood_);
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
    UpdateBlock& powers, const size_t burnin)
{
  SusceptibilityMRW* update = new SusceptibilityMRW(tag, params, powers,
      burnin, *random_, likelihood_);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an InfectivityMRW updater onto the MCMC stack
InfectivityMRW*
Mcmc::NewInfectivityMRW(const string tag, UpdateBlock& params,
    UpdateBlock& powers, const size_t burnin)
{
  InfectivityMRW* update = new InfectivityMRW(tag, params, powers, burnin,
      *random_, likelihood_);
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

//! Sets the number of infection time (and occult) updates per sweep of the MCMC
void
Mcmc::setNumIUpdates(const size_t n)
{
  numIUpdates_ = n;
}




bool
Mcmc::UpdateI()
{

  size_t index = random_->integer(likelihood_.GetNumInfecs());
  double newIN = random_->extreme(a, b); // Independence sampler
  double oldIN = likelihood_.GetIN(index);

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Moving '" << it->getId() << "' from " << it->getI() << " to "
        << newI << endl;
#endif

  float piCur = likelihood_.GetCurrentValue();
  float piCan = likelihood_.UpdateI(index, newIN);

  if (!likelihood_.IsInfecDC(index))
    { // Known infection
      piCan += log(extremepdf(newIN, a, b));
      piCur += log(extremepdf(oldIN, a, b));
    }
  else
    { // Occult
      piCan += log(1 - extremecdf(newIN, a, b));
      piCur += log(1 - extremecdf(oldIN, a, b));
    }

  double qRatio = log(extremecdf(oldIN, a, b) / extremecdf(
      newIN, a, b));

  double accept = piCan - piCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "ACCEPT" << endl;
#endif
      // Update the infection
      likelihood_.Accept();
      return true;
    }
  else
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "REJECT" << endl;
#endif
      likelihood_.Reject();
      return false;
    }
}

bool
Mcmc::AddI()
{
  size_t numSusceptible = likelihood_.GetNumPossibleOccults();

  if (numSusceptible == 0)
    return false;

  size_t index = random_->integer(numSusceptible);

  double inProp = random_->gaussianTail(-(1.0 / b), 1.0 / (a * b * b));

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Adding '" << it->getId() << "' at " << newI << endl;
#endif

  double logPiCur = likelihood_.GetCurrentValue();

  double logPiCan = likelihood_.UpdateI(index, inProp) + log(1.0 - extremecdf(inProp, a, b));
  double qRatio = log((1.0 / (likelihood_.GetNumOccults() + 1))
      / ((1.0 / numSusceptible) * gaussianTailPdf(inProp, -1.0 / b, 1.0 / (a
          * b * b))));

  double accept = logPiCan - logPiCur + qRatio;

  // Perform accept/reject step.
  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
        cerr << "ACCEPT" << endl;
#endif
      likelihood_.Accept();
      return true;
    }
  else
    {
#ifndef NDEBUG
        cerr << "REJECT" << endl;
#endif
     likelihood_.Reject();
      return false;
    }
}

bool
Mcmc::DeleteI()
{
  if (likelihood_.GetNumOccults() == 0)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << __FUNCTION__ << endl;
      if (mpirank_ == 0)
        cerr << "Occults empty. Not deleting" << endl;
#endif
      return false;
    }

  size_t numSusceptible = likelihood_.GetNumPossibleOccults();

  size_t toRemove = random_->integer(likelihood_.GetNumOccults());

  float inTime = likelihood_.GetIN(likelihood_.GetNumKnownInfecs() + toRemove);
  float logPiCur = likelihood_.GetCurrentValue() + log(1 - extremecdf(inTime, a, b));

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Deleting '" << it->getId() << "'" << endl;
#endif

  float logPiCan = likelihood_.DeleteI(toRemove);
  double qRatio = log((1.0 / (numSusceptible + 1) * gaussianTailPdf(inTime,
      -1.0 / b, 1.0 / (a * b * b))) / (1.0 / likelihood_.GetNumOccults()));

  // Perform accept/reject step.
  double accept = logPiCan - logPiCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "ACCEPT" << endl;
#endif
      likelihood_.Accept();
      return true;
    }
  else
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "REJECT" << endl;
#endif
      likelihood_.Reject();
      return false;
    }
}

void
Mcmc::Update()
{
  // Performs a sweep over the parameters
  //  this could be generalized!

  try
    {

      for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin(); it
      != updateStack_.end(); ++it)
        it->Update();

          for (size_t infec = 0; infec < numIUpdates_; ++infec)
            {
              size_t pickMove = random_->integer(1);
              switch (pickMove)
                {
              case 0:
                acceptMove_ += UpdateI();
                callsMove_++;
                break;
              case 1:
                acceptAdd_ += AddI();
                callsAdd_++;
                break;
              case 2:
                acceptDel_ += DeleteI();
                callsDel_++;
                break;
              default:
                throw logic_error("Unknown move!");
                }
            }
    }
  catch (logic_error& e)
    {
      cerr << "Logic Error occurred: " << e.what() << endl;
    }
  catch (...)
  {
      cerr << "Unknown error in " << __FILE__ << ":" << __LINE__ << endl;
  }
}


map<string, float>
Mcmc::GetAcceptance() const
{
  map<string, float> acceptance;
  for (boost::ptr_list<McmcUpdate>::const_iterator it = updateStack_.begin(); it
      != updateStack_.end(); ++it)
    {
      acceptance[it->GetTag()] = it->GetAcceptance();
    }

  acceptance["UpdateI"] = (float)acceptMove_ / (float)callsMove_;
  acceptance["AddI"] = (float)acceptAdd_ / (float)callsAdd_;
  acceptance["DelI"] = (float)acceptDel_ / (float)callsDel_;

  return acceptance;
}

void
Mcmc::ResetAcceptance()
{
  for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin(); it
    != updateStack_.end(); ++it) it->ResetAcceptance();

  acceptMove_ = 0; callsMove_ = 0;
  acceptDel_ = 0;  callsDel_ = 0;
  acceptAdd_ = 0;  callsAdd_ = 0;
}

