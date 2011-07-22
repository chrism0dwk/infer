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

//#ifdef __LINUX__
//#include <acml_mv.h>
//#define log fastlog
//#define exp fastexp
//#endif

#include "Mcmc.hpp"

using namespace EpiRisk;

// Constants
const double a = 0.015;
const double b = 0.8;
const double tuneI = 2.0;
const double numIUpdates = 10;

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
  double dist = sqrt(dx * dx + dy * dy);

  return dist;
}

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& transParams,
    Parameters& detectParams, const size_t randomSeed) :
  pop_(population), txparams_(transParams), dxparams_(detectParams),
      integPressTime_(1.0), meanParams_(transParams), dicUpdates_(0),
      postMeanDev_(0.0)
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
      processInfectives_.push_back(j);
      pos += mpiprocs_;
      advance(j, mpiprocs_);
    }

  // Calculate log likelihood
  calcLogLikelihood(logLikelihood_);
  all_reduce(comm_, logLikelihood_.local, logLikelihood_.global,
      plus<double> ());
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
    size_t burnin)
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
    std::vector<double>& alpha)
{
  SpeciesMRW* update = new SpeciesMRW(tag, params, alpha, 1, *random_,
      logLikelihood_, this);
  updateStack_.push_back(update);

  return update;
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
  double infectivity = //i.getCovariates().cattle + txparams_(4)*i.getCovariates().pigs + txparams_(5)*i.getCovariates().sheep;
      pow(txparams_(4) * i.getCovariates().cattle, txparams_(7)) + txparams_(5)
          * pow(i.getCovariates().pigs, txparams_(8)) + txparams_(6) * pow(
          i.getCovariates().sheep, txparams_(9));
  return infectivity;
}

inline
double
Mcmc::susceptibility(const Population<TestCovars>::Individual& i,
    const Population<TestCovars>::Individual& j) const
{
  double susceptibility = //j.getCovariates().cattle + txparams_(9)*j.getCovariates().pigs + txparams_(10)*j.getCovariates().sheep;
      pow(txparams_(10) * j.getCovariates().cattle / 10, txparams_(13))
          + txparams_(11) * pow(j.getCovariates().pigs, txparams_(14))
          + txparams_(12) * pow(j.getCovariates().sheep, txparams_(15));

  return susceptibility;
}

inline
double
Mcmc::beta(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x, i.getCovariates().y,
      j.getCovariates().x, j.getCovariates().y);
  if (distance <= 25.0)
    {
      return txparams_(0) * infectivity(i, j) * susceptibility(i, j)
          * txparams_(2) / (txparams_(2) * txparams_(2) + distance * distance);
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
  if (distance <= 25.0)
    {
      return txparams_(0) * txparams_(1) * infectivity(i, j) * susceptibility(
          i, j) * txparams_(2) / (txparams_(2) * txparams_(2) + distance
          * distance);
    }
  else
    return 0.0;
}

inline
double
Mcmc::instantPressureOn(const Population<TestCovars>::InfectiveIterator& j,
    const double Ij)
{
  if (Ij <= pop_.infecBegin()->getI())
    return 1.0; // Return 1 if j is I1

  double sumPressure = 0.0;
  Population<TestCovars>::InfectiveIterator i = pop_.infecBegin();
  Population<TestCovars>::InfectiveIterator stop = pop_.infecLowerBound(Ij); // Don't need people infected after me.

  while (i != stop)
    {
      if (i != j)
        { // Skip i==j
          if (i->getN() > Ij)
            {
              sumPressure += beta(*i, *j);
            }
          else if (i->getR() > Ij)
            {
              sumPressure += betastar(*i, *j);
            }
        }
      ++i;
    }
  sumPressure += txparams_(3);

  return sumPressure;
}

inline
double
Mcmc::integPressureOn(const Population<TestCovars>::PopulationIterator& j,
    const double Ij)
{
  double integPressure = 0.0;
  double I1 = min(Ij, pop_.infecBegin()->getI());
  Population<TestCovars>::InfectiveIterator infj = pop_.asI(j);
  Population<TestCovars>::InfectiveIterator stop = pop_.infecLowerBound(Ij);
  for (Population<TestCovars>::InfectiveIterator i = pop_.infecBegin(); i
      != stop; // Don't need people infected after k
  ++i)
    {
      if (i == infj)
        continue; // Don't add pressure to ourselves
      // Infective -> Susceptible pressure
      integPressure += beta(*i, *j) * (min(min(i->getN(), pop_.getObsTime()),
          Ij) - min(i->getI(), Ij));

      // Notified -> Susceptible pressure
      integPressure += betastar(*i, *j) * (min(
          min(i->getR(), pop_.getObsTime()), Ij) - min(min(i->getN(),
          pop_.getObsTime()), Ij));
    }

  integPressure += txparams_(3) * (min(Ij, pop_.getObsTime()) - I1);
  return -integPressure;
}

void
Mcmc::calcLogLikelihood(Likelihood& logLikelihood)
{
  // Calculates log likelihood
  ProcessInfectives::const_iterator j = processInfectives_.begin();
  Population<TestCovars>::InfectiveIterator stop;
  logLikelihood.local = 0.0;
  size_t numInfecs = pop_.numInfected();

  logLikelihood.productCache.clear();

  // First calculate the log product (happens on master node)
  while (j != processInfectives_.end())
    {
      double tmp = instantPressureOn(*j, (*j)->getI());
      logLikelihood.productCache.insert(make_pair((*j)->getId(), tmp));
      logLikelihood.local += log(tmp);
      ++j;
    }

  //Now calculate the integral
  size_t pos;
  Population<TestCovars>::PopulationIterator k;
  for (pos = mpirank_, k = pop_.begin() + mpirank_; pos < pop_.size(); pos
      += mpiprocs_, k += mpiprocs_)
    {
      logLikelihood.local += integPressureOn(k, k->getI());
    }

  all_reduce(comm_, logLikelihood.local, logLikelihood.global, plus<double> ());
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

void
Mcmc::updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
    const double newTime, Likelihood& updatedLogLik)
{
  // Calculates an updated likelihood for an infection time move
  double logLikelihood = logLikelihood_.local;
  Population<TestCovars>::PopulationIterator popj = pop_.asPop(j);
  updatedLogLik.productCache.clear();

  // Sort out I1
  if (j == pop_.infecBegin() or newTime < pop_.infecBegin()->getI())
    {
      double oldTime = j->getI();
      pop_.moveInfectionTime(j, newTime);
      calcLogLikelihood(updatedLogLik);
      pop_.moveInfectionTime(j, oldTime);
      return;
    }

  // Product part of likelihood
  for (ProcessInfectives::const_iterator i = processInfectives_.begin(); i
      != processInfectives_.end(); ++i)
    {
      double myPressure = 0.0;
      // Subtract old pressure
      map<string, double>::const_iterator myProduct =
          logLikelihood_.productCache.find((*i)->getId());
      if (myProduct != logLikelihood_.productCache.end())
        {
          myPressure = myProduct->second;
          logLikelihood -= log(myPressure);
        }

      if (*i == j) // Instantaneous pressure on j
        {
          // Product part of likelihood
          if (newTime == POSINF)
            myPressure = 0.0;
          else
            myPressure = instantPressureOn(*i, newTime);
        }
      else
        {
          if (j->isIAt((*i)->getI()))
            myPressure -= beta(*j, **i);

          if (newTime <= (*i)->getI() && (*i)->getI() < j->getN())
            myPressure += beta(*j, **i);
        }

      if (myPressure != 0.0)
        { // Zero pressure individual occurs when an occult is deleted
          updatedLogLik.productCache.insert(
              make_pair((*i)->getId(), myPressure));
          logLikelihood += log(myPressure);
        }
    }

  // Integral part of likelihood
  size_t pos;
  Population<TestCovars>::PopulationIterator i;
  for (pos = mpirank_, i = pop_.begin() + mpirank_; pos < pop_.size(); pos
      += mpiprocs_, i += mpiprocs_)
    {
      if (i == popj)
        {
          logLikelihood -= integPressureOn(i, i->getI());
          logLikelihood += integPressureOn(i, newTime);
        }
      else
        {
          double myBeta = beta(*j, *i);
          logLikelihood += myBeta * (min(min(j->getN(), pop_.getObsTime()),
              i->getI()) - min(min(j->getI(), pop_.getObsTime()), i->getI()));
          logLikelihood -= myBeta * (min(min(j->getN(), pop_.getObsTime()),
              i->getI()) - min(min(newTime, pop_.getObsTime()), i->getI()));
        }
    }
  cout << "Process " << mpirank_ << " finished with loglikelihood = "
      << logLikelihood << endl;
  updatedLogLik.local = logLikelihood;

  all_reduce(comm_, logLikelihood, updatedLogLik.global, plus<double> ());
  cout << "all_reduce completed with logLikelihood = " << updatedLogLik.global
      << endl;
  //  std::vector<double> components;
  //  gather(comm_,updatedLogLik.local,components,0);
  //  cout << "gather completed" << endl;
  //  if(mpirank_ == 0) {
  //      cout << "Components:";
  //      for(std::vector<double>::const_iterator it = components.begin();
  //          it != components.end();
  //          it++)
  //        {
  //          cout << " " << *it;
  //        }
  //      cout << endl;
  //  }
}

bool
Mcmc::updateI(const size_t index)
{
  cout << __FUNCTION__ << endl;
  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

//  if (it->getN() > pop_.getObsTime())
//    {
//      cout << "Rejecting move occult '" << it->getId() << "' (" << index
//          << "), N(" << it->getN() << ") > " << pop_.getObsTime() << endl;
//      ;
//      return false;
//    }

  double effectiveN = min(it->getN(), pop_.getObsTime());
  double newI = effectiveN - (effectiveN - it->getI()) * exp(random_->gaussian(
      0, tuneI));
  //double newI = it->getN() - random_->extreme(a, b); // Independence sampler

  cout << "Moving '" << it->getId() << "' (" << index << "): " << it->getI()
      << " -> " << newI << endl;

  if (index == 0 or newI < pop_.I1().getI())
    return false;

  Likelihood logLikCan;
  updateIlogLikelihood(it, newI, logLikCan);
  if (logLikCan.global != logLikCan.global)
    cout << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;

  double piCan = logLikCan.global;
  double piCur = logLikelihood_.global;

  if (it->getN() <= pop_.getObsTime())
    { // Known infection
      piCan += log(extremepdf(it->getN() - newI, a, b));
      piCur += log(extremepdf(it->getN() - it->getI(), a, b));
    }
  else
    { // Occult
      piCan += log(1 - extremecdf(pop_.getObsTime() - newI, a, b));
      piCur += log(1 - extremecdf(pop_.getObsTime() - it->getI(), a, b));
    }

  double qRatio = log((it->getN() - newI) / (it->getN() - it->getI()));
  double accept = piCan - piCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
      // Update the infection
      pop_.moveInfectionTime(it, newI);
      logLikelihood_ = logLikCan;
      return true;
    }
  else
    {
      return false;
    }
}

bool
Mcmc::addI()
{
  cout << __FUNCTION__ << endl;
  cout.precision(20);
  size_t numSusceptible = pop_.numSusceptible();
  cout << "Loglikelihood: " << logLikelihood_.global << endl;
  Population<TestCovars>::InfectiveIterator it = pop_.infecEnd();
  advance(it, random_->integer(numSusceptible));

  double inProp = random_->gaussianTail(-(1.0 / b), 1.0 / (a * b * b));
  double newI = pop_.getObsTime() - inProp;

  cout << "Adding '" << it->getId() << "' at I=" << newI << endl;

  double logPiCur = logLikelihood_.global;

  // Which process will calculate the addition?
  bool myAddition = false;
  if (mpirank_ == random_->integer(comm_.size()))
    myAddition = true;

  //Add to processInfectives and occultList
  if (myAddition)
    processInfectives_.push_back(it);

  cout.precision(20);
  Likelihood logLikCan;
  updateIlogLikelihood(it, newI, logLikCan);
  if (logLikCan.global != logLikCan.global)
    cout << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;
  double logPiCan = logLikCan.global + log(1.0 - extremecdf(inProp, a, b));
  double qRatio = log((1.0 / (occultList_.size() + 1))
      / ((1.0 / numSusceptible) * gaussianTailPdf(inProp, -1.0 / b, 1.0 / (a
          * b * b))));

  double accept = logPiCan - logPiCur + qRatio;

  // Perform accept/reject step.
  if (log(random_->uniform()) < accept)
    {
      cout << "ACCEPT ADDITION" << endl;
      pop_.moveInfectionTime(it, newI);
      occultList_.push_back(it);
      logLikelihood_ = logLikCan;
      return true;
    }
  else
    {
      cout << "REJECT ADDITION" << endl;
      // Delete from processInfectives and occultList
      if (myAddition)
        processInfectives_.erase(find(processInfectives_.begin(),
            processInfectives_.end(), it));
      return false;
    }
}

bool
Mcmc::deleteI()
{
  cout << __FUNCTION__ << endl;
  size_t numSusceptible = pop_.numSusceptible();
  cout << "Loglikelihood: " << logLikelihood_.global << endl;
  if (occultList_.empty())
    {
      cout << "Occults empty. Not deleting" << endl;
      return false;
    }

  ProcessInfectives::const_iterator idx = occultList_.begin();
  advance(idx, random_->integer(occultList_.size()));

  Population<TestCovars>::InfectiveIterator it = *idx;
  cout << "Deleting '" << it->getId() << "'" << endl;
  double logPiCur = logLikelihood_.global + log(1 - extremecdf(
      pop_.getObsTime() - it->getI(), a, b));

  cout.precision(20);
  Likelihood logLikCan;
  updateIlogLikelihood(it, POSINF, logLikCan);
  if (logLikCan.global != logLikCan.global)
    cout << "NAN in log likelihood (" << __FUNCTION__ << ")" << endl;
  double logPiCan = logLikCan.global;
  double qRatio = log((1.0 / (numSusceptible + 1) * gaussianTailPdf(
      pop_.getObsTime() - it->getI(), -1 / b, 1 / (a * b * b))) / (1.0
      / occultList_.size()));

  // Perform accept/reject step.
  double accept = logPiCan - logPiCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
      pop_.moveInfectionTime(it, POSINF);
      logLikelihood_ = logLikCan;
      cout << "ACCEPT DELETION" << endl;
      // Delete from processInfectives and occultList
      ProcessInfectives::iterator toErase = find(processInfectives_.begin(),
          processInfectives_.end(), it);
      if (toErase != processInfectives_.end())
        processInfectives_.erase(toErase);
      occultList_.erase(find(occultList_.begin(), occultList_.end(), it));
      return true;
    }
  else
    {
      cout << "REJECT DELETION" << endl;
      return false;
    }
}

map<string, double>
Mcmc::run(const size_t numIterations,
    McmcWriter<Population<TestCovars> >& writer)
{
  // Runs the MCMC


  cout << "Starting with " << pop_.numSusceptible() << " susceptibles and "
      << pop_.numInfected() << " infectives" << endl;
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
          if (k % 1 == 0)
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

          for (size_t infec = 0; infec < numIUpdates; ++infec)
            {
              cout << "Picked: ";
              size_t pickMove = random_->integer(3);
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
              //checkInfecOrder();
              //checkProcPopConsistency();
              cout << "Num occults = " << occultList_.size() << endl;
              cout << "gLogLikelihood: " << logLikelihood_.global << endl;
            }

          updateDIC();

          txparams_(16) = getMeanI2N();
          txparams_(17) = getMeanOccI();

          // Update the adaptive mcmc
          if (mpirank_ == 0)
            {
              writer.write(pop_);
              writer.write(txparams_);
            }
        }

    }
  catch (logic_error& e)
    {
      cout << "Logic Error occurred: " << e.what() << endl;
      return acceptance;
    }
  if (mpirank_ >= 0)
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
      acceptance["I"] /= (numIterations * numIUpdates / 3);
      acceptance["add"] /= (numIterations * numIUpdates / 2);
      acceptance["delete"] /= (numIterations * numIUpdates / 2);
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
          cout << "======================RANK " << proc
              << "====================" << endl;
          map<string, double>::const_iterator it =
              logLikelihood_.productCache.begin();
          cout << "ID \t \t Cache \t \t TmpCache \t \t Difference\n" << endl;
          ;
          MPI::COMM_WORLD.Barrier();
          while (it != logLikelihood_.productCache.end())
            {
              cout << it->first << ":\t" << it->second << "\t" << endl;
              MPI::COMM_WORLD.Barrier();
              ++it;
            }
          cout
              << "==============================================================\n"
              << endl;
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
              cout << "Corrupted population at ID !" << it->getId() << endl << endl << endl;
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
  for (Population<TestCovars>::InfectiveIterator it = ++pop_.infecBegin();
      it != pop_.infecEnd();
      it++)
    {
      if (prevTime > it->getI()) {
          cout << "Found out-of-sequence index at ID " << it->getId() << endl;
          pop_.dumpInfected();
          throw logic_error("Out of order index!!");
      }
      prevTime = it->getI();
    }
}
