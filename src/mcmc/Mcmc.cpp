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

#ifdef __LINUX__
#include <acml_mv.h>
#define log fastlog
#define exp fastexp
#endif

#include "Mcmc.hpp"

using namespace EpiRisk;

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

  // MPI setup
  int largc = 1;
  char** largv;

  mpirank_ = comm_.rank();
  mpiprocs_ = comm_.size();

  // Random number generation
  random_ = new Random(randomSeed);

  //AdMCMC covariance.
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

  logLikelihood_ = calcLogLikelihood();
  all_reduce(comm_, logLikelihood_, gLogLikelihood_, plus<double> ());
  productCache_ = productCacheTmp_;
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
  if (distance <= 10)
    {
      return params_(0) * exp(-params_(2) * (distance - 5));
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
  if (distance <= 10)
    return params_(1) * exp(-params_(2) * (distance - 5));
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

  ProcessInfectives::const_iterator j = processInfectives_.begin();
  Population<TestCovars>::InfectiveIterator stop;
  double logLikelihood = 0.0;
  size_t numInfecs = pop_.numInfected();
  size_t pos = 0;

  productCacheTmp_.clear();

  // First calculate the log product (happens on master node)
  while (j != processInfectives_.end())
    {
      double tmp = instantPressureOn(*j, (*j)->getI());
      productCacheTmp_.insert(make_pair((*j)->getId(), tmp));
      logLikelihood += log(tmp);
      ++j;
    }

  //Now calculate the integral
  Population<TestCovars>::PopulationIterator k = pop_.begin();
  pos = mpirank_;
  advance(k, mpirank_);
  while (pos < pop_.size())
    {
      logLikelihood += integPressureOn(k, k->getI());
      pos += mpiprocs_;
      advance(k, mpiprocs_);
    }

  return logLikelihood;
}

double
Mcmc::updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
    const double newTime)
{
  // Calculates an updated likelihood for an infection time move

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
      return logLikelihood;
    }

  // Product part of likelihood
  for (ProcessInfectives::const_iterator i = processInfectives_.begin(); i
      != processInfectives_.end(); ++i)
    {
      double myPressure = productCache_.find((*i)->getId())->second;
      logLikelihood -= log(myPressure);

      if (*i == j) // Instantaneous pressure on j
        {
          // Product part of likelihood
          myPressure = instantPressureOn(*i, newTime);
        }
      else
        {
          if (j->isIAt((*i)->getI()))
            myPressure -= beta(*j, **i);

          if (newTime <= (*i)->getI() && (*i)->getI() < j->getN())
            myPressure += beta(*j, **i);
        }

      productCacheTmp_.insert(make_pair((*i)->getId(), myPressure));
      logLikelihood += log(myPressure);
    }

  // Integral part of likelihood
  double pos = mpirank_;
  Population<TestCovars>::PopulationIterator i = pop_.begin();
  advance(i, mpirank_);
  while (pos < pop_.size())
    {
      if (i == popj)
        {
          logLikelihood -= integPressureOn(i, i->getI());
          logLikelihood += integPressureOn(i, newTime);
        }
      else
        {
          double myBeta = beta(*j, *i);
          logLikelihood += myBeta * (min(j->getN(), i->getI()) - min(j->getI(),
              i->getI()));
          logLikelihood -= myBeta * (min(j->getN(), i->getI()) - min(newTime,
              i->getI()));
        }
      advance(i, mpiprocs_);
      pos += mpiprocs_;
    }

  return logLikelihood;
}

bool
Mcmc::updateTrans()
{
  Parameters oldParams = params_;
  Random::Variates logvars;
  double logPiCur;
  double gLogLikCan;

  logPiCur = gLogLikelihood_;
  logPiCur += log(params_(0).prior());
  logPiCur += log(params_(1).prior());
  logPiCur += log(params_(2).prior());
  logPiCur += log(params_(3).prior());

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

  params_(0) *= exp(logvars[0]);
  params_(1) *= exp(logvars[1]);
  params_(2) *= exp(logvars[2]);
  params_(3) *= exp(logvars[3]);

  double logLikCan = calcLogLikelihood();
  all_reduce(comm_, logLikCan, gLogLikCan, plus<double> ());

  double logPiCan = gLogLikCan;
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
  if (log(random_->uniform()) < accept)
    {
      logLikelihood_ = logLikCan;
      gLogLikelihood_ = gLogLikCan;
      productCache_ = productCacheTmp_;
      return true;
    }
  else
    {
      params_(0) = oldParams(0);
      params_(1) = oldParams(1);
      params_(2) = oldParams(2);
      params_(3) = oldParams(3);
      return false;
    }

}

bool
Mcmc::updateI(const size_t index)
{
  double newI;
  double gLogLikCan;
  IProposal proposal;

  proposal.index = index;
  proposal.I = random_->extreme(0.015, 0.8);

  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, proposal.index);
  newI = it->getN() - proposal.I;

  double logLikCan = updateIlogLikelihood(it, newI);
  all_reduce(comm_, logLikCan, gLogLikCan, plus<double> ());

  double a = gLogLikCan - gLogLikelihood_;

  if (log(random_->uniform()) < a)
    {
      // Update the infection
      pop_.moveInfectionTime(it, newI);
      logLikelihood_ = logLikCan;
      gLogLikelihood_ = gLogLikCan;
      productCache_ = productCacheTmp_;
      return true;
    }
  else
    {
      return false;
    }

}

map<string, double>
Mcmc::run(const size_t numIterations,
    McmcWriter<Population<TestCovars> >& writer)
{
  // Runs the MCMC

  map<string, double> acceptance;
  int toMove;

  if (mpirank_ == 0)
    {
      writer.open();
    }

  acceptance["transParms"] = 0.0;
  acceptance["I"] = 0.0;

  for (size_t k = 0; k < numIterations; ++k)
    {
      if (mpirank_ == 0 && k % 50 == 0)
        cout << "Iteration " << k << endl;

      acceptance["transParms"] += updateTrans();

      for (size_t infec = 0; infec < 20; ++infec)
        {
          toMove = random_->integer(pop_.numInfected());
          acceptance["I"] += updateI(toMove);
        }

      // Update the adaptive mcmc
      logTransCovar_->sample();
      if (mpirank_ == 0)
        {
          writer.write(pop_);
          writer.write(params_);
        }
    }

  if (mpirank_ == 0)
    {
      cout << "\n";
      cout << logTransCovar_->getCovariance() << endl;

      writer.close();
      acceptance["transParms"] /= numIterations;
      acceptance["I"] /= (numIterations * 20);
    }
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
  for (size_t proc = 0; proc < mpiprocs_; ++proc)
    {
      MPI::COMM_WORLD.Barrier();
      if (mpirank_ == proc)
        {
          cout << "======================RANK " << proc
              << "====================" << endl;
          map<string, double>::const_iterator it = productCache_.begin();
          cout << "ID \t \t Cache \t \t TmpCache \t \t Difference\n" << endl;
          ;
          MPI::COMM_WORLD.Barrier();
          while (it != productCache_.end())
            {
              cout << it->first << ":\t" << it->second << "\t"
                  << productCacheTmp_[it->first] << "\t"
                  << productCacheTmp_[it->first] - it->second << endl;
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
