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
#include <acml_mv.h>

#include "Mcmc.hpp"

using namespace EpiRisk;

inline
double
dist(const double x1, const double y1, const double x2, const double y2)
{
  double dx = (x1 - x2);
  double dy = (y1 - y2);
  double dist = sqrt(dx * dx + dy * dy);

  return dist;
}

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& parameters,
    const size_t randomSeed) :
  pop_(population), params_(parameters), logLikelihood_(0)
{
  random_ = new Random(randomSeed);
  EmpCovar<ExpTransform>::CovMatrix initCov(params_.size());
  for (size_t i = 0; i < params_.size(); ++i)
    {
      for (size_t j = 0; j < params_.size(); ++j)
        {
          if (i == j)
            initCov(i, j) = 0.00001;
          else
            initCov(i, j) = 0.0;
        }
    }

  logTransCovar_ = new EmpCovar<LogTransform> (params_, initCov);
  stdCov_ = new ublas::matrix<double>(params_.size(), params_.size());

  for (size_t i = 0; i < params_.size(); ++i)
    {
      for (size_t j = 0; j < params_.size(); ++j)
        {
          if (i == j)
            (*stdCov_)(i, j) = 0.1 / params_.size();
          else
            (*stdCov_)(i, j) = 0.0;
        }

    }

  if(false) pop_.dumpInfected();
  logLikelihood_ = calcLogLikelihood();
}

Mcmc::~Mcmc()
{
  // Nothing to do at present
  delete stdCov_;
  delete random_;
  delete logTransCovar_;
}

double
Mcmc::getLogLikelihood() const
{
  return logLikelihood_;
}

inline
double
Mcmc::beta(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x, i.getCovariates().y,
      j.getCovariates().x, j.getCovariates().y);
  if (distance <= 25)
    {
      return params_(0) * fastexp(-params_(2) * (distance - 5));
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
  if (distance <= 25)
    return params_(1) * fastexp(-params_(2) * (distance - 5));
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
  sumPressure += params_(3);
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
      integPressure += beta(*i, *j) * (min(i->getN(), Ij) - min(i->getI(), Ij));

      // Notified -> Susceptible pressure
      integPressure += betastar(*i, *j) * (min(i->getR(), Ij) - min(i->getN(),
          Ij));
    }

  integPressure += params_(3) * (min(Ij, pop_.getObsTime()) - I1);

  return -integPressure;
}

double
Mcmc::calcLogLikelihood()
{
  // Calculates log likelihood

  Population<TestCovars>::InfectiveIterator j = pop_.infecBegin();
  Population<TestCovars>::InfectiveIterator stop;
  double logLikelihood = 0.0;

  productCache_.clear();
  productCache_.insert(make_pair(j->getId(), 1.0));

  // First calculate the log product
  ++j; // Don't include I1
  while (j != pop_.infecEnd())
    {
      double tmp = instantPressureOn(j, j->getI());
      productCache_.insert(make_pair(j->getId(), tmp));
      logLikelihood += log(tmp);
      ++j;
    }

  productCacheTmp_ = productCache_;
  // Now calculate the integral
    for (Population<TestCovars>::PopulationIterator j = pop_.begin(); j
        != pop_.end(); ++j)
      {
        logLikelihood += integPressureOn(j, j->getI());
      }

  return logLikelihood;
}

double
Mcmc::updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
    const double newTime)
{
  // Calculates an updated likelihood for an infection time move
  cout << "LogLik before: " << logLikelihood_ << endl;
  double logLikelihood = logLikelihood_;
  Population<TestCovars>::PopulationIterator popj = pop_.asPop(j);
  productCacheTmp_.clear();

  // Sort out I1
  if (j == pop_.infecBegin() or newTime < pop_.infecBegin()->getI())
    {
      double oldTime = j->getI();
      pop_.moveInfectionTime(j, newTime);
      logLikelihood = calcLogLikelihood();
      pop_.moveInfectionTime(j, oldTime);
      productCacheTmp_ = productCache_;
      return logLikelihood;
    }

  // First instantaneous pressure on j
  double myPressure = productCache_.find(j->getId())->second;
  logLikelihood -= fastlog(myPressure);

  myPressure = instantPressureOn(j, newTime);
  productCacheTmp_.insert(make_pair(j->getId(), myPressure));
  logLikelihood += fastlog(myPressure);

  logLikelihood -= integPressureOn(popj, j->getI());
  logLikelihood += integPressureOn(popj, newTime);

  //Pressure from j on i
  for (Population<TestCovars>::InfectiveIterator i = pop_.infecBegin(); i
      != pop_.infecEnd(); ++i)
    {
      if (i == j)
        continue;

      // Product first
      myPressure = productCache_.find(i->getId())->second;
      logLikelihood -= fastlog(myPressure);

      if (j->isIAt(i->getI()))
        myPressure -= beta(*j, *i);

      if (newTime <= i->getI() && i->getI() < j->getN())
        myPressure += beta(*j, *i);

      logLikelihood += fastlog(myPressure);
      productCacheTmp_.insert(make_pair(i->getId(), myPressure));
    }

   //Integral now
    for (Population<TestCovars>::PopulationIterator i = pop_.begin(); i
        != pop_.end(); ++i)
      {
        if (i == popj) continue;
        double myBeta = beta(*j, *i);
        logLikelihood += myBeta * (min(j->getN(), i->getI()) - min(
            j->getI(), i->getI()));
        logLikelihood -= myBeta * (min(j->getN(), i->getI()) - min(newTime,
            i->getI()));
      }
  cout << "LogLik after: " << logLikelihood << endl;
  return logLikelihood;
}

bool
Mcmc::updateTrans()
{
  double logPiCur = logLikelihood_;
  logPiCur += log(params_(0).prior());
  logPiCur += log(params_(1).prior());
  logPiCur += log(params_(2).prior());
  logPiCur += log(params_(3).prior());

  Parameters oldParams = params_;
  Random::Variates logvars;

  if (random_->uniform() < 0.95)
    {
      try
        {
          logvars = random_->mvgauss(logTransCovar_->getCovariance() * 2.38
              * 2.38 / params_.size());
        }
      catch (cholesky_error& e)
        {
          logvars = random_->mvgauss(*stdCov_);
        }
    }
  else
    logvars = random_->mvgauss(*stdCov_);

  params_(0) *= exp(logvars(0));
  params_(1) *= exp(logvars(1));
  params_(2) *= exp(logvars(2));
  params_(3) *= exp(logvars(3));

  dumpParms();

  double logLikCan = calcLogLikelihood();
  double logPiCan = logLikCan;
  logPiCan += log(params_(0).prior());
  logPiCan += log(params_(1).prior());
  logPiCan += log(params_(2).prior());
  logPiCan += log(params_(3).prior());

  double qRatio = 0.0;
  qRatio += log(params_(0) / oldParams(0));
  qRatio += log(params_(1) / oldParams(1));
  qRatio += log(params_(2) / oldParams(2));
  qRatio += log(params_(3) / oldParams(3));

  double accept = logPiCan - logPiCur + qRatio;
  return false;
  if (log(random_->uniform()) < accept)
    {
      logLikelihood_ = logLikCan;
      return true;
    }
  else
    {
      params_ = oldParams;
      return false;
    }

}

bool
Mcmc::updateI(const size_t index)
{
  if (index == 0 ) return false;
  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

  cout << "Moving " << index << ", id = " << it->getId() << ", I = " << it->getI() << endl;

  double newI = it->getI();//random_->gamma(4, 1);

  double logLikCan = updateIlogLikelihood(it, newI);
  double a = logLikCan - logLikelihood_;
  return false;
  if ( fastlog(random_->uniform()) < a)
    {
      pop_.moveInfectionTime(it,newI);
      logLikelihood_ = logLikCan;
      productCache_ = productCacheTmp_;
      return true;
    }
  else return false;
}

map<string, double>
Mcmc::run(const size_t numIterations,
    McmcWriter<Population<TestCovars> >& writer)
{
  // Runs the MCMC

  map<string, double> acceptance;
  acceptance["transParms"] = 0.0;
  acceptance["I"] = 0.0;
  for (size_t k = 0; k < numIterations; ++k)
    {
      //if (k % 50 == 0)
        cout << "\rIteration " << k << flush;
      acceptance["transParms"] += updateTrans();
      cout << "Parms: " << logLikelihood_ << endl;

      for(size_t infec=0; infec < 1; ++infec) {
          //acceptance["I"] += updateI(random_->integer(pop_.numInfected()));
          cout << "Infec time: " << logLikelihood_ << endl;
      }

      // Update the adaptive mcmc
      logTransCovar_->sample();

      writer.write(pop_);
      writer.write(params_);
    }
  cout << "\n";
  cout << logTransCovar_->getCovariance() << endl;
  dumpParms();
  //pop_.dumpInfected();

  //acceptance["transParms"] /= numIterations;
  //acceptance["I"] /= (numIterations);
  return acceptance;
}

void
Mcmc::dumpParms() const
{
  ublas::vector<Parameter>::iterator it;
  it = params_.begin();
  while (it != params_.end())
    {
      cout << *it << " ";
      ++it;
    }
  cout << endl;
}

void
Mcmc::dumpProdCache()
{
  map<string, double>::const_iterator it = productCache_.begin();
  cout << "ID \t \t Cache \t \t TmpCache \t \t Difference\n\n";
  while (it != productCache_.end())
    {
      cout << it->first << ":\t" << it->second << "\t"
          << productCacheTmp_[it->first] << "\t" << productCacheTmp_[it->first]
          - it->second << endl;
      ++it;
    }
}
