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
#include <boost/mpi/datatype.hpp>
#include <boost/mpi/collectives.hpp>
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

inline
double
dist(const double x1, const double y1, const double x2, const double y2)
{
  double dx = (x1 - x2);
  double dy = (y1 - y2);
  double dist = dx * dx + dy * dy;

  return dist;
}

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& transParams,
    Parameters& detectParams, const size_t randomSeed) :
  pop_(population), txparams_(transParams), dxparams_(detectParams),
      integPressTime_(1.0), meanParams_(transParams), dicUpdates_(0),
      postMeanDev_(0.0), numIUpdates_(0), timeCalc_(0.0), timeUpdate_(0.0),
      numCalc_(0), numUpdate_(0)
{

  // MPI setup
  int largc = 1;
  char** largv;

  mpirank_ = comm_.rank();
  mpiprocs_ = comm_.size();

  // Random number generation
  random_ = new Random(randomSeed);

  // Set up load balancing
  //  for(size_t p=0; p<mpiprocs_;++p) elements_.push_back(0);
  //  loadBalance();

  // Set up process-bound infectives
  Population<TestCovars>::InfectiveIterator j = pop_.infecBegin();
  size_t pos = mpirank_;
  advance(j, mpirank_);
  while (pos < pop_.numInfected())
    {
      processInfectives_.insert(j);
      pos += mpiprocs_;
      advance(j, mpiprocs_);
    }

  // Set up DC list and powerrs of cattle, pigs, and sheep
  for (Population<TestCovars>::PopulationIterator it = pop_.begin(); it
      != pop_.end(); it++)
    {
      // DC List
      if (it->getN() < POSINF and it->getI() == POSINF)
        dcList_.insert(pop_.asI(it));

      // Powers
      const_cast<TestCovars&> (it->getCovariates()).cattleinf = pow(
          it->getCovariates().cattle, txparams_(10));
      const_cast<TestCovars&> (it->getCovariates()).pigsinf = pow(
          it->getCovariates().pigs, txparams_(11));
      const_cast<TestCovars&> (it->getCovariates()).sheepinf = pow(
          it->getCovariates().sheep, txparams_(12));
      const_cast<TestCovars&> (it->getCovariates()).cattlesusc = pow(
          it->getCovariates().cattle, txparams_(13));
      const_cast<TestCovars&> (it->getCovariates()).pigssusc = pow(
          it->getCovariates().pigs, txparams_(14));
      const_cast<TestCovars&> (it->getCovariates()).sheepsusc = pow(
          it->getCovariates().sheep, txparams_(15));
    }

  // Calculate log likelihood
  calcLogLikelihood(logLikelihood_);
  testLik_ = logLikelihood_;
  if (mpirank_ == 0)
    cout << "Log likelihood starts at " << logLikelihood_.global << std::endl;

  // Set up DIC
  initializeDIC();
}

Mcmc::~Mcmc()
{
  delete random_;
}

//! Pushes an updater onto the MCMC stack
SingleSiteLogMRW*
Mcmc::newSingleSiteLogMRW(Parameter& param, const double tuning)
{
  SingleSiteLogMRW* update = new SingleSiteLogMRW(param.getTag(), param,
      tuning, *random_, logLikelihood_, this);
  updateStack_.push_back(update);
}

//! Pushes an updater onto the MCMC stack
AdaptiveMultiLogMRW*
Mcmc::newAdaptiveMultiLogMRW(const string name, UpdateBlock& updateGroup,
    size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<LogTransform>::CovMatrix initCov(updateGroup.size());
  for (size_t i = 0; i < updateGroup.size(); ++i)
    for (size_t j = 0; j <= i; ++j)
      initCov(i, j) = i == j ? 0.1 : 0.0;

  AdaptiveMultiLogMRW* update = new AdaptiveMultiLogMRW(name, updateGroup,
      burnin, *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an updater onto the MCMC stack
AdaptiveMultiMRW*
Mcmc::newAdaptiveMultiMRW(const string name, UpdateBlock& updateGroup,
    const size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<Identity>::CovMatrix initCov(updateGroup.size());
  for (size_t i = 0; i < updateGroup.size(); ++i)
    for (size_t j = 0; j <= i; ++j)
      initCov(i, j) = i == j ? 0.1 : 0.0;

  AdaptiveMultiMRW* update = new AdaptiveMultiMRW(name, updateGroup, burnin,
      *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an SpeciesMRW updater onto the MCMC stack
SpeciesMRW*
Mcmc::newSpeciesMRW(const string tag, UpdateBlock& params,
    std::vector<double>& alpha, const size_t burnin)
{
  SpeciesMRW* update = new SpeciesMRW(tag, params, alpha, burnin, *random_,
      logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an SusceptibilityMRW updater onto the MCMC stack
SusceptibilityMRW*
Mcmc::newSusceptibilityMRW(const string tag, UpdateBlock& params,
    UpdateBlock& powers, const size_t burnin)
{
  SusceptibilityMRW* update = new SusceptibilityMRW(tag, params, powers,
      burnin, *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an InfectivityMRW updater onto the MCMC stack
InfectivityMRW*
Mcmc::newInfectivityMRW(const string tag, UpdateBlock& params,
    UpdateBlock& powers, const size_t burnin)
{
  InfectivityMRW* update = new InfectivityMRW(tag, params, powers, burnin,
      *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes as InfectivityPowMRW onto the MCMC stack
InfectivityPowMRW*
Mcmc::newInfectivityPowMRW(const string tag, UpdateBlock& params,
    const size_t burnin)
{
  InfectivityPowMRW* update = new InfectivityPowMRW(tag, params, burnin,
      *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes a SusceptibilityPowMRW onto the MCMC stack
SusceptibilityPowMRW*
Mcmc::newSusceptibilityPowMRW(const string tag, UpdateBlock& params,
    const size_t burnin)
{
  SusceptibilityPowMRW* update = new SusceptibilityPowMRW(tag, params, burnin,
      *random_, logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes a SellkeSerializer onto the MCMC stack
SellkeSerializer*
Mcmc::newSellkeSerializer(const string filename)
{
  SellkeSerializer* update = new SellkeSerializer(filename, *random_,
      logLikelihood_, this);
  updateStack_.push_back(update);
  return update;
}

//! Sets the number of infection time (and occult) updates per sweep of the MCMC
void
Mcmc::setNumIUpdates(const size_t n)
{
  numIUpdates_ = n;
}

double
Mcmc::getLogLikelihood() const
{
  return logLikelihood_.global;
}

inline
double
Mcmc::infectivity(const Population<TestCovars>::Individual& i,
    const Population<TestCovars>::Individual& j) const
{
  double infectivity = txparams_(4) * i.getCovariates().cattleinf //pow(i.getCovariates().cattle,txparams_(10))
      + txparams_(5) * i.getCovariates().pigsinf //pow(i.getCovariates().pigs, txparams_(11))
      + txparams_(6) * i.getCovariates().sheepinf; //pow(i.getCovariates().sheep, txparams_(12));

  return infectivity;
}

inline
double
Mcmc::susceptibility(const Population<TestCovars>::Individual& i,
    const Population<TestCovars>::Individual& j) const
{

  double susceptibility = txparams_(7) * j.getCovariates().cattlesusc
      + txparams_(8) * j.getCovariates().pigssusc + txparams_(9)
      * j.getCovariates().sheepsusc;

  return susceptibility;
}

inline
double
Mcmc::beta(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x, i.getCovariates().y,
      j.getCovariates().x, j.getCovariates().y);
  if (distance <= 25.0 * 25.0)
    {
      return txparams_(0) * infectivity(i, j) * txparams_(2) / (txparams_(2)
          * txparams_(2) + distance);
    }
  else
    return 0.0;
}

inline
double
Mcmc::betastar(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x, i.getCovariates().y,
      j.getCovariates().x, j.getCovariates().y);
  if (distance <= 25.0 * 25.0)
    {
      return txparams_(0) * txparams_(1) * infectivity(i, j) * txparams_(2)
          / (txparams_(2) * txparams_(2) + distance);
    }
  else
    return 0.0;
}

inline
double
Mcmc::instantPressureOn(const Population<TestCovars>::InfectiveIterator& j,
    const double Ij)
{
  //if (Ij <= pop_.infecBegin()->getI())
  //  return 1.0; // Return 1 if j is I1

  double sumPressure = 0.0;
  size_t j_idx = distance(pop_.asPop(pop_.infecBegin()),pop_.asPop(j));
  Population<TestCovars>::Individual::ConnectionList::const_iterator i =
      j->getConnectionList().begin();
  while (i != j->getConnectionList().end() and pop_[*i].getI() < Ij)
    {
      if (*i != j_idx)
        { // Skip i==j
          if (pop_[*i].getN() >= Ij)
            {
              sumPressure += beta(pop_[*i], *j);
            }
          else if (pop_[*i].getR() >= Ij)
            {
              sumPressure += betastar(pop_[*i], *j);
            }
        }
      ++i;

    }

  sumPressure *= susceptibility(pop_.I1(), *j);
  sumPressure += txparams_(3);

  return sumPressure;
}

inline
double
Mcmc::integPressureOn(const Population<TestCovars>::PopulationIterator& j,
    const double Ij)
{
  double integPressure = 0.0;

  double I1 = pop_.infecBegin()->getI(); //min(Ij, pop_.infecBegin()->getI());

  // Get the latest time that j can possibly be susceptible
  double jMaxSuscepTime = min(min(Ij, pop_.getObsTime()), j->getN());
  size_t j_idx = distance(pop_.asPop(pop_.infecBegin()),j);
  Population<TestCovars>::Individual::ConnectionList::const_iterator i =
      j->getConnectionList().begin();

  while (i != j->getConnectionList().end() and pop_[*i].getI() < Ij)
    {
      if (*i != j_idx)
        {
          // Infective -> Susceptible pressure
          integPressure += beta(pop_[*i], *j) * (min(pop_[*i].getN(), jMaxSuscepTime)
              - min(pop_[*i].getI(), jMaxSuscepTime));

          // Notified -> Susceptible pressure
          integPressure += betastar(pop_[*i], *j) * (min(pop_[*i].getR(),
              jMaxSuscepTime) - min(pop_[*i].getN(), jMaxSuscepTime));
        }
      ++i;
    }
  integPressure *= susceptibility(pop_.I1(), *j);

  return integPressure;
}

void
Mcmc::calcLogLikelihood(Likelihood& logLikelihood)
{

  timeval start, end;
  gettimeofday(&start, NULL);

  // Calculates log likelihood

  logLikelihood.local = 0.0;
  size_t numInfecs = pop_.numInfected();

  logLikelihood.productCache.clear();

  // First calculate the log product
  // Don't calculate for I1
  for (ProcessInfectives::const_iterator j = processInfectives_.begin(); j
      != processInfectives_.end(); ++j)
    {
      if (*j == pop_.infecBegin())
        continue; // Don't add a product entry for I1
      double tmp = instantPressureOn(*j, (*j)->getI());
      logLikelihood.productCache.insert(make_pair((*j)->getId(), tmp));
      logLikelihood.local += log(tmp);
    }

  //Now calculate the integral
  logLikelihood.integPressure.clear();
  size_t pos;
  Population<TestCovars>::PopulationIterator k;
  for (pos = mpirank_, k = pop_.begin() + mpirank_; pos < pop_.size(); pos
      += mpiprocs_, k += mpiprocs_)
    {
      double tmp = integPressureOn(k, k->getI());
      tmp += txparams_(3) * (min(k->getI(), pop_.getObsTime())
          - pop_.I1().getI());
      logLikelihood.integPressure.insert(make_pair(k->getId(), tmp));
      logLikelihood.local -= tmp; // Require minus the integral
    }

  all_reduce(comm_, logLikelihood.local, logLikelihood.global, plus<double> ());
  gettimeofday(&end, NULL);
  numCalc_++;
  timeCalc_ = onlineMean(timeinseconds(start, end), timeCalc_, numCalc_);

}

/////////DIC Methods -- probably should be in a separate class//////////
void
Mcmc::initializeDIC()
{
  // Parameter means
  meanParams_ = txparams_;

  // Infection time means
  ProcessInfectives::iterator it;
  for (it = processInfectives_.begin(); it != processInfectives_.end(); it++)
    {
      meanInfecTimes_.insert(make_pair((*it)->getId(), 0.0));
    }
  dicUpdates_ = 0;
}

void
Mcmc::updateDIC()
{
  // Updates DIC

  dicUpdates_++;

  // Mean posterior deviance
  double delta = -2 * getLogLikelihood() - postMeanDev_;
  postMeanDev_ += delta / dicUpdates_;

  // Mean posterior parameters
  for (size_t i = 0; i < txparams_.size(); i++)
    {
      delta = txparams_(i) - meanParams_(i);
      meanParams_(i) += delta / dicUpdates_;
    }

  // Mean infection times
  map<string, double>::iterator it;
  for (it = meanInfecTimes_.begin(); it != meanInfecTimes_.end(); it++)
    {
      delta = pop_.getById(it->first).getI() - it->second; //SLOW!
      it->second += delta / dicUpdates_;
    }
}

DIC
Mcmc::getDIC()
{
  // Returns a struct containing the DIC

  // First calculate the deviance at the posterior means:
  Parameters oldParams = txparams_;
  txparams_ = meanParams_;
  map<string, double> oldInfecTimes;
  for (ProcessInfectives::iterator it = processInfectives_.begin(); it
      != processInfectives_.end(); it++)
    {
      Population<TestCovars>::InfectiveIterator i = *it;
      oldInfecTimes.insert(make_pair(i->getId(), i->getI()));
      pop_.moveInfectionTime(i, meanInfecTimes_[i->getId()]);
    }

  Likelihood myLikelihood;
  calcLogLikelihood(myLikelihood);

  // Assemble DIC
  DIC rv;
  rv.Dbar = postMeanDev_;
  rv.Dhat = -2 * myLikelihood.global;
  rv.pD = rv.Dbar - rv.Dhat;
  rv.DIC = rv.Dbar - rv.pD;

  // Undo changes
  txparams_ = oldParams;
  for (ProcessInfectives::iterator it = processInfectives_.begin(); it
      != processInfectives_.end(); it++)
    {
      pop_.moveInfectionTime(*it, oldInfecTimes[(*it)->getId()]);
    }

  return rv;
}

double
Mcmc::getMeanI2N() const
{
  double sumI2Nlocal = 0.0;
  for (ProcessInfectives::const_iterator it = processInfectives_.begin(); it
      != processInfectives_.end(); it++)
    {
      if ((*it)->getN() <= pop_.getObsTime())
        sumI2Nlocal += (*it)->getN() - (*it)->getI();
    }
  double sumI2Nglobal = 0.0;
  all_reduce(comm_, sumI2Nlocal, sumI2Nglobal, plus<double> ());

  return sumI2Nglobal / (pop_.numInfected() - occultList_.size());
}

double
Mcmc::getMeanOccI() const
{
  double sum = 0.0;
  for (ProcessInfectives::const_iterator it = occultList_.begin(); it
      != occultList_.end(); it++)
    {
      sum += (*it)->getI();
    }
  return sum / occultList_.size();
}

//void
//Mcmc::newUpdateIlogLikelihood(
//    const Population<TestCovars>::InfectiveIterator& j, const double newTime,
//    Likelihood& updatedLogLik)
//{
//  timeval start, end;
//  gettimeofday(&start, NULL);
//
//  const Population<TestCovars>::InfectiveIterator& I1(pop_.infecBegin());
//  const Population<TestCovars>::InfectiveIterator& I2(++pop_.infecBegin());
//
//  // Calculates an updated likelihood for an infection time move
//  double logLikelihood = 0.0; //logLikelihood_.local;
//  updatedLogLik.productCache = logLikelihood_.productCache;
//
//  // Adjust pressure on the movee
//  if (processInfectives_.count(j) != 0)
//    { // Node which has the movee does the calculation -- not optimal!
//
//      // Adjust product first
//      double myPressure = 0.0;
//
//      map<string, double>::const_iterator myProduct =
//          logLikelihood_.productCache.find(j->getId());
//      if (myProduct != logLikelihood_.productCache.end())
//        {
//          myPressure = myProduct->second;
//          logLikelihood -= log(myPressure);
//        }
//
//      if (newTime < pop_.getObsTime())
//        {
//          myPressure = instantPressureOn(j, newTime);
//          updatedLogLik.productCache[j->getId()] = myPressure; // Automatically adds j to the productCache if it didn't exist before (ie Adding)
//          logLikelihood += log(myPressure);
//        }
//      else
//        {
//          updatedLogLik.productCache.erase(j->getId());
//        }
//
//      // Now adjust integral
//      double press;
//      press = integPressureOn(pop_.asPop(j), j->getI());
//      logLikelihood += press;
//      press = integPressureOn(pop_.asPop(j), newTime);
//      logLikelihood -= press;
//    }
//
//  // Adjust product part -- iterate over connection list and update pressure on
//  //                        local process's infectives.
//  Population<TestCovars>::Individual::ConnectionList::const_iterator i;
//  for (i = j->getConnectionList().begin(); i != j->getConnectionList().end(); i++)
//    {
//      if (processInfectives_.count(pop_.asI(pop_[*i])) != 0)
//        {
//
//          // Skip if i == I1 -- no pressure change
//          if (*i == &(pop_.I1()))
//            continue;
//
//          // Product First
//          // Fetch cached pressure
//          double myPressure = 0.0;
//          map<string, double>::const_iterator myProduct =
//              logLikelihood_.productCache.find((*i)->getId());
//          if (myProduct != logLikelihood_.productCache.end())
//            {
//              myPressure = myProduct->second;
//              logLikelihood -= log(myPressure);
//            }
//
//          // Adjust pressure - subtract old pressure
//          if (j->isIAt((*i)->getI()))
//            myPressure -= beta(*j, **i) * susceptibility(*j, **i);
//          else if (j->isNAt((*i)->getI()) and j->getI() < pop_.getObsTime()
//              and newTime > pop_.getObsTime())
//            myPressure -= betastar(*j, **i) * susceptibility(*j, **i);
//
//          // Adjust pressure - add new pressure
//          if (newTime < (*i)->getI() && (*i)->getI() <= j->getN())
//            myPressure += beta(*j, **i) * susceptibility(*j, **i);
//          else if (j->isNAt((*i)->getI()) and j->getI() > pop_.getObsTime()
//              and newTime < pop_.getObsTime())
//            myPressure += betastar(*j, **i) * susceptibility(*j, **i);
//
//          updatedLogLik.productCache[(*i)->getId()] = myPressure;
//          logLikelihood += log(myPressure);
//          if (logLikelihood == NEGINF)
//            cerr << "Log Likelihood neg inf for individual " << (*i)->getId()
//                << " with pressure " << myPressure << endl;
//        }
//    }
//
//  // Adjust the integral part
//  size_t pos = 0;
//  for (i = j->getConnectionList().begin() + mpirank_, pos = mpirank_; pos
//      < j->getConnectionList().size(); i += mpiprocs_, pos += mpiprocs_)
//    {
//      double myBeta = beta(*j, **i) * susceptibility(*j, **i);
//      double iMaxSuscTime = min(min((*i)->getI(), (*i)->getN()),
//          pop_.getObsTime());
//      logLikelihood += myBeta * (min(j->getN(), iMaxSuscTime) - min(min(
//          j->getI(), j->getN()), iMaxSuscTime));
//
//      if (j->getI() < pop_.getObsTime() and newTime > pop_.getObsTime()) // Remove betastar if j is not infectious anymore
//        logLikelihood += betastar(*j, **i) * susceptibility(*j, **i) * (min(
//            j->getR(), iMaxSuscTime) - min(j->getN(), iMaxSuscTime));
//
//      logLikelihood -= myBeta * (min(j->getN(), iMaxSuscTime) - min(min(
//          newTime, j->getN()), iMaxSuscTime));
//
//      if (j->getI() > pop_.getObsTime() and newTime < pop_.getObsTime()) // Add betastar is j is now infectious
//        logLikelihood -= betastar(*j, **i) * susceptibility(*j, **i) * (min(
//            j->getR(), iMaxSuscTime) - min(j->getN(), iMaxSuscTime));
//    }
//
//  // Background integrated pressure, and bits to sort out a change of I1
//  if (j == I1)
//    {
//      map<string, double>::iterator myProduct;
//
//      if (newTime < I2->getI())
//        {
//#ifndef NDEBUG
//          cerr << "I1 -> I1 <<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//#endif
//          myProduct = updatedLogLik.productCache.find(I1->getId());
//        }
//      else
//        {
//#ifndef NDEBUG
//          cerr << "I1 -> I* <<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//#endif
//          myProduct = updatedLogLik.productCache.find(I2->getId());
//        }
//
//      // Delete pressure on new I1
//      if (myProduct != updatedLogLik.productCache.end())
//        {
//          logLikelihood -= log(myProduct->second);
//          updatedLogLik.productCache.erase(myProduct);
//        }
//
//      // Update background integrated pressure
//      if (processInfectives_.count(j) != 0)
//        {
//          logLikelihood -= txparams_(3) * (newTime - min(newTime, I2->getI())); // Add bgPressure to I1 if I1 -> I*
//          double bgPress = txparams_(3) * (pop_.I1().getI() - min(newTime,
//              I2->getI()));
//          logLikelihood -= bgPress * (pop_.size() - 1); // Add bgPressure to all others
//          cerr << "Adjust background integral from I1 move: +" << bgPress
//              * (pop_.size() - 1);
//        }
//
//    }
//
//  // I* -> I1 or S -> I1
//  else if (j != I1 and newTime < I1->getI())
//    {
//#ifndef NDEBUG
//      cerr << "I* -> I1 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
//#endif
//      // Delete pressure on j -- it is now I1
//      map<string, double>::iterator myProduct =
//          updatedLogLik.productCache.find(j->getId());
//      if (myProduct != updatedLogLik.productCache.end())
//        {
//          logLikelihood -= log(myProduct->second);
//          updatedLogLik.productCache.erase(myProduct);
//        }
//
//      // Add pressure to the old I1
//      if (processInfectives_.count(I1) != 0)
//        {
//          double myPressure = txparams_(3);
//
//          if (find(j->getConnectionList().begin(),
//              j->getConnectionList().end(), &(*I1))
//              != j->getConnectionList().end())
//            {
//              if (I1->getI() <= j->getN())
//                myPressure += beta(*j, *I1) * susceptibility(*j, *I1);
//              else if (j->isNAt(I1->getI()))
//                myPressure += betastar(*j, *I1) * susceptibility(*j, *I1);
//            }
//          updatedLogLik.productCache[I1->getId()] = myPressure;
//          logLikelihood += log(myPressure);
//
//          // Update background integrated pressure
//          logLikelihood += txparams_(3) * (j->getI() - pop_.I1().getI()); // Subtract old background pressure from the new I1
//          logLikelihood -= txparams_(3) * (pop_.I1().getI() - newTime)
//              * (pop_.size() - 1.0); // Add pressure integral to all others
//        }
//
//    }
//
//  else
//    // Update background pressure integral
//    if (processInfectives_.count(I1) != 0) logLikelihood -= txparams_(3) * (newTime - j->getI());
//
//  updatedLogLik.local = logLikelihood + logLikelihood_.local;
//
//  all_reduce(comm_, updatedLogLik.local, updatedLogLik.global, plus<double> ());
//
//  gettimeofday(&end, NULL);
//  numUpdate_++;
//  timeUpdate_ = onlineMean(timeinseconds(start, end), timeUpdate_, numUpdate_);
//}

bool
Mcmc::updateI(const size_t index)
{
  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

  double effectiveN = min(it->getN(), pop_.getObsTime());
  //double newI = effectiveN - (effectiveN - it->getI()) * exp(random_->gaussian(
  //    0, tuneI));
  double newI = effectiveN - random_->extreme(a, b); // Independence sampler

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Moving '" << it->getId() << "' from " << it->getI() << " to "
        << newI << endl;
#endif

  Likelihood logLikCan;
  //newUpdateIlogLikelihood(it, newI, logLikCan);

#ifndef NDEBUG
  Likelihood tmp;
  double oldI = it->getI();
  pop_.moveInfectionTime(it, newI);
  calcLogLikelihood(tmp);
  pop_.moveInfectionTime(it, oldI);

  if (fabs(logLikCan.global - tmp.global) > 1e-11 and mpirank_ == 0)
    {
      size_t rank = distance(pop_.infecBegin(), it);
      cerr.precision(20);
      cerr << "Log likelihood error in " << __FUNCTION__ << " for individual "
          << it->getId() << " (I=" << it->getI() << ", I*=" << newI
          << ", rank=" << rank << ")  Update=" << logLikCan.global << ", Full="
          << tmp.global << endl;
    }
  if (logLikCan.global != logLikCan.global)
    cout << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;
#endif

  double piCan = logLikCan.global;
  double piCur = logLikelihood_.global;

  if (occultList_.count(it) == 0)
    { // Known infection
      piCan += log(extremepdf(effectiveN - newI, a, b));
      piCur += log(extremepdf(effectiveN - it->getI(), a, b));
    }
  else
    { // Occult
      piCan += log(1 - extremecdf(effectiveN - newI, a, b));
      piCur += log(1 - extremecdf(effectiveN - it->getI(), a, b));
    }

  double qRatio = log(extremecdf(effectiveN - it->getI(), a, b) / extremecdf(
      effectiveN - newI, a, b));
  //log((effectiveN - newI) / (effectiveN - it->getI()));
  double accept = piCan - piCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "ACCEPT" << endl;
#endif
      // Update the infection
      pop_.moveInfectionTime(it, newI);
      logLikelihood_ = logLikCan;
      testLik_ = logLikCan;
      return true;
    }
  else
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "REJECT" << endl;
#endif
      return false;
    }
}

bool
Mcmc::addI()
{
  size_t numSusceptible = dcList_.size();//pop_.numSusceptible();//
  //Population<TestCovars>::InfectiveIterator it = pop_.infecEnd();
  //advance(it, random_->integer(numSusceptible));

  if (numSusceptible == 0)
    return false;
  ProcessInfectives::iterator pi = dcList_.begin();
  advance(pi, random_->integer(numSusceptible));
  Population<TestCovars>::InfectiveIterator it = *pi;

  double inProp = random_->gaussianTail(-(1.0 / b), 1.0 / (a * b * b));
  double newI = min(pop_.getObsTime(), it->getN()) - inProp;

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Adding '" << it->getId() << "' at " << newI << endl;
#endif

  double logPiCur = logLikelihood_.global;

  //Add to processInfectives for a random processor
  if (mpirank_ == random_->integer(comm_.size()))
    processInfectives_.insert(it);

  Likelihood logLikCan;
  //newUpdateIlogLikelihood(it, newI, logLikCan);

#ifndef NDEBUG
  //  Likelihood tmp;
  //  double oldI = it->getI();
  //  pop_.moveInfectionTime(it, newI);
  //  calcLogLikelihood(tmp);
  //  pop_.moveInfectionTime(it, oldI);
  //
  //  if(fabs(logLikCan.global - tmp.global) > 1e-11 and mpirank_ == 0) {
  //      size_t rank = distance(pop_.infecBegin(),it);
  //      cerr.precision(20);
  //      cerr << "Log likelihood error in " << __FUNCTION__ << " for individual " << it->getId() << " (I=" << it->getI() << ", I*=" << newI << ", rank=" << rank << ")  Update=" << logLikCan.global << ", Full=" << tmp.global << endl;
  //  }
  if (logLikCan.global != logLikCan.global)
    if (mpirank_ == 0)
      cerr << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;
#endif

  double logPiCan = logLikCan.global + log(1.0 - extremecdf(inProp, a, b));
  double qRatio = log((1.0 / (occultList_.size() + 1))
      / ((1.0 / numSusceptible) * gaussianTailPdf(inProp, -1.0 / b, 1.0 / (a
          * b * b))));

  double accept = logPiCan - logPiCur + qRatio;

  // Perform accept/reject step.
  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "ACCEPT" << endl;
#endif
      pop_.moveInfectionTime(it, newI);
      dcList_.erase(it);
      occultList_.insert(it);
      logLikelihood_ = logLikCan;
      return true;
    }
  else
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "REJECT" << endl;
#endif

      // Delete from processInfectives and occultList
      processInfectives_.erase(it);
      return false;
    }
}

bool
Mcmc::deleteI()
{
  size_t numSusceptible = dcList_.size();//pop_.numSusceptible();//
  if (occultList_.empty())
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << __FUNCTION__ << endl;
      if (mpirank_ == 0)
        cerr << "Occults empty. Not deleting" << endl;
#endif
      return false;
    }

  ProcessInfectives::const_iterator idx = occultList_.begin();
  advance(idx, random_->integer(occultList_.size()));

  Population<TestCovars>::InfectiveIterator it = *idx;
  double inTime = min(pop_.getObsTime(), it->getN()) - it->getI();
  double logPiCur = logLikelihood_.global + log(1 - extremecdf(inTime, a, b));

#ifndef NDEBUG
  if (mpirank_ == 0)
    cerr << "Deleting '" << it->getId() << "'" << endl;
#endif

  Likelihood logLikCan;
  //newUpdateIlogLikelihood(it, POSINF, logLikCan);

#ifndef NDEBUG
  //  Likelihood tmp;
  //  double oldI = it->getI();
  //  pop_.moveInfectionTime(it, POSINF);
  //  processInfectives_.erase(it);
  //  calcLogLikelihood(tmp);
  //  processInfectives_.insert(it);
  //  pop_.moveInfectionTime(it, oldI);
  //
  //  if(fabs(logLikCan.global - tmp.global) > 1e-11 and mpirank_ == 0) {
  //      size_t rank = distance(pop_.infecBegin(), it);
  //      cerr.precision(20);
  //      cerr << "Log likelihood error in " << __FUNCTION__ << " for individual " << it->getId() << " (I=" << it->getI() << ", I*=" << POSINF << ", rank=" << rank << ")  Update=" << logLikCan.global << ", Full=" << tmp.global << endl;
  //  }
  if (logLikCan.global != logLikCan.global)
    if (mpirank_ == 0)
      cout << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;
#endif

  double logPiCan = logLikCan.global;
  double qRatio = log((1.0 / (numSusceptible + 1) * gaussianTailPdf(inTime,
      -1.0 / b, 1.0 / (a * b * b))) / (1.0 / occultList_.size()));

  // Perform accept/reject step.
  double accept = logPiCan - logPiCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "ACCEPT" << endl;
#endif
      pop_.moveInfectionTime(it, POSINF);
      logLikelihood_ = logLikCan;
      // Delete from processInfectives and occultList
      processInfectives_.erase(it);
      dcList_.insert(it);
      occultList_.erase(it);
      return true;
    }
  else
    {
#ifndef NDEBUG
      if (mpirank_ == 0)
        cerr << "REJECT" << endl;
#endif
      return false;
    }
}

map<string, double>
Mcmc::run(const size_t numIterations,
    McmcWriter<Population<TestCovars> >& writer)
{
  // Runs the MCMC


#ifndef NDEBUG
  cout << "Starting with " << pop_.numSusceptible() << " susceptibles and "
      << pop_.numInfected() << " infectives" << endl;
#endif

  map<string, double> acceptance;
  int toMove;

  if (mpirank_ == 0)
    {
      writer.open(txparams_);
      writer.write(pop_);
      writer.write(txparams_);
    }

  acceptance["I"] = 0.0;
  acceptance["add"] = 0.0;
  acceptance["delete"] = 0.0;

  try
    {
      for (size_t k = 0; k < numIterations; ++k)
        {
          if (k % 100 == 0 and mpirank_ == 0)
            cout << "Iteration " << k << endl;

          //          if (k % 100 == 0)
          //            {
          //              DIC myDIC = getDIC();
          //              if (mpirank_ >= 0)
          //                {
          //                  cout << "=======DIC=======\n" << "Dbar: " << myDIC.Dbar
          //                      << "\n" << "Dhat: " << myDIC.Dhat << "\n" << "pD: "
          //                      << myDIC.pD << "\n" << "DIC: " << myDIC.DIC << endl;
          //                }
          //            }


          for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin(); it
              != updateStack_.end(); ++it)
            it->update();

          for (size_t infec = 0; infec < numIUpdates_; ++infec)
            {
              size_t pickMove = random_->integer(1);
              switch (pickMove)
                {
              case 0:
                toMove = random_->integer(pop_.numInfected());
                acceptance["I"] += updateI(toMove);
                break;
              case 1:
                acceptance["add"] += addI();
                break;
              case 2:
                acceptance["delete"] += deleteI();
                break;
              default:
                throw logic_error("Unknown move!");
                }
            }
          //#ifndef NDEBUG
          cerr.precision(15);
          if (mpirank_ == 0) {
              cerr << "Current I1 = " << pop_.I1().getId() << " at "
              << pop_.I1().getI() << endl;
              cerr << "Current I2 = " << (++pop_.infecBegin())->getId()
              << " at " << (++pop_.infecBegin())->getI() << endl;
          }
          Likelihood tmp;
          calcLogLikelihood(tmp);
          if (mpirank_ == 0)
            {
          cerr << "Calc log likelihood = " << tmp.global << " ("
              << tmp.productCache.size() << ")" << endl;
          if (fabs(logLikelihood_.global - tmp.global) > 10e-4)
            cerr << "Likelihoods not equal: " << tmp.global << "\t"
                << logLikelihood_.global << "\t***" << endl;
            }
          logLikelihood_ = tmp;
          if (mpirank_ == 0) cerr << "Num current infections = " << pop_.numInfected() << endl;
          //#endif
          cout.precision(15);
          if (mpirank_ == 0)
            cout << "Log likelihood = " << logLikelihood_.global << endl;
          updateDIC();

          txparams_(16) = getMeanI2N();
          txparams_(17) = getMeanOccI();
          txparams_(18) = logLikelihood_.global;

          // Update the adaptive mcmc
          if (mpirank_ == 0)
            {
              writer.write(pop_);
              writer.write(txparams_);
            }
        }

      if (mpirank_ == 0)
        {
          cerr << "Mean time for full likelihood calculation: " << timeCalc_
              << endl;
          cerr << "Mean time for likelihood update: " << timeUpdate_ << endl;
        }

    }
  catch (logic_error& e)
    {
      cout << "Logic Error occurred: " << e.what() << endl;
      return acceptance;
    }
  if (mpirank_ == 0)
    {
      cout << "\n";
      writer.close();
      cout << "Acceptances:\n";
      cout << "============\n";
      for (boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin(); it
          != updateStack_.end(); ++it)
        {
          cout << it->getTag() << ": " << it->getAcceptance() << "\n";
        }
      acceptance["I"] /= (numIterations * numIUpdates_ / 3.0);
      acceptance["add"] /= (numIterations * numIUpdates_ / 3.0);
      acceptance["delete"] /= (numIterations * numIUpdates_ / 3.0);
    }
  return acceptance;

}

void
Mcmc::dumpParms() const
{
  ublas::vector<Parameter>::iterator it;
  it = txparams_.begin();
  while (it != txparams_.end())
    {
      cout << *it << " ";
      ++it;
    }
  cout << endl;
}

void
Mcmc::dumpProdCache()
{
  for (size_t proc = 0; proc < mpiprocs_; ++proc)
    {
      MPI::COMM_WORLD.Barrier();
      if (mpirank_ == proc)
        {
          cerr << "======================RANK " << proc
              << "====================" << endl;
          map<string, double>::const_iterator it =
              logLikelihood_.productCache.begin();
          cerr << "ID \t \t Cache \t \t TmpCache \t \t Difference\n" << endl;
          ;
          MPI::COMM_WORLD.Barrier();
          while (it != logLikelihood_.productCache.end())
            {
              cerr << it->first << ":\t" << it->second << "\t" << endl;
              MPI::COMM_WORLD.Barrier();
              ++it;
            }
          cerr
              << "==============================================================\n"
              << "Length = " << logLikelihood_.productCache.size() << endl;
        }
      MPI::COMM_WORLD.Barrier();
    }
}

void
Mcmc::loadBalance()
{
  std::vector<double> times;
  multimap<double, size_t> processors; // ordered by ascending speed
  mpi::all_gather(comm_, integPressTime_, times);

  if (mpirank_ == 0)
    {
      cout << "Times: ";
      for (size_t i = 0; i < times.size(); ++i)
        cout << times[i] << " ";
      cout << endl;
    }

  double timesSum = 0.0;
  for (size_t i = 0; i < times.size(); ++i)
    {
      times[i] = 1 / times[i];
      processors.insert(make_pair(times[i], i));
      timesSum += times[i];
    }

  int leftover = 0;
  for (size_t i = 0; i < times.size(); ++i)
    {
      times[i] = times[i] / timesSum * pop_.size();
      elements_[i] = round(times[i]);
      leftover += elements_[i];
    }

  leftover = pop_.size() - leftover;

  if (leftover < 0) // Remove elements from slowest processors
    {
      map<double, size_t>::const_iterator it = processors.begin();
      while (leftover != 0)
        {
          elements_[it->second] -= 1;
          leftover += 1;
          it++;
        }
    }
  else if (leftover > 0) // Add elements to fastest processors
    {
      map<double, size_t>::const_iterator it = processors.end();
      while (leftover != 0)
        {
          it--;
          elements_[it->second] += 1;
          leftover -= 1;
        }
    }

  if (mpirank_ == 0)
    {
      cout << "Elements: ";
      for (size_t i = 0; i < elements_.size(); ++i)
        cout << elements_[i] << " ";
      cout << endl;
    }

}

void
Mcmc::checkProcPopConsistency()
{
  cout.precision(15);
  for (Population<TestCovars>::InfectiveIterator it = pop_.infecBegin(); it
      != pop_.infecEnd(); it++)
    {
      std::vector<string> ids;
      all_gather(comm_, it->getId(), ids);
      for (int i = 1; i < comm_.size(); i++)
        {
          if (ids[0] != ids.at(i))
            {
              cout << "Corrupted population at ID !" << it->getId() << endl
                  << endl << endl;
              pop_.dumpInfected();
              throw logic_error("Corrupted population!");
            }
        }
    }
}

void
Mcmc::checkInfecOrder()
{
  cout.precision(15);
  double prevTime = pop_.I1().getI();
  for (Population<TestCovars>::InfectiveIterator it = ++pop_.infecBegin(); it
      != pop_.infecEnd(); it++)
    {
      if (prevTime > it->getI())
        {
          cout << "Found out-of-sequence index at ID " << it->getId() << endl;
          pop_.dumpInfected();
          throw logic_error("Out of order index!!");
        }
      prevTime = it->getI();
    }
}
