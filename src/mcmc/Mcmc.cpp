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
#include <mpi.h>

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

  mpirank_ = MPI::COMM_WORLD.Get_rank();
  mpiprocs_ = MPI::COMM_WORLD.Get_size();

  // Random number generator

  if (mpirank_ == 0)
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

      if (false)
        pop_.dumpInfected();
    }

  gLogLikelihood_ = calcLogLikelihood();
  logLikelihood_ = logLikCan_;
  logProduct_ = logProductCan_;
}

Mcmc::~Mcmc()
{
  // Nothing to do at present
  if (mpirank_ == 0)
    {
      delete stdCov_;
      delete random_;
      delete logTransCovar_;
    }
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

  Population<TestCovars>::InfectiveIterator j = pop_.infecBegin();
  Population<TestCovars>::InfectiveIterator stop;
  double logLikelihood = 0.0;
  size_t numInfecs = pop_.numInfected();

  // Advance the iterator to our starting position
  advance(j, mpirank_);
  size_t pos = mpirank_;

  // First calculate the log product (happens on master node)
  if (j == pop_.infecBegin())
    {
      pos += mpiprocs_;
      if (pos < numInfecs) advance(j,mpiprocs_);
    }
  while (pos < numInfecs)
    {
      double tmp = instantPressureOn(j, j->getI());
      logLikelihood += log(tmp);
      pos += mpiprocs_;
      advance(j, mpiprocs_);
    }

  logProductCan_ = logLikelihood;

  // Now calculate the integral
  Population<TestCovars>::PopulationIterator k = pop_.begin();
  pos = mpirank_;
  advance(k, mpirank_);
  while (pos < pop_.size())
    {
      logLikelihood += integPressureOn(k, k->getI());
      pos += mpiprocs_;
      advance(k, mpiprocs_);
    }

  logLikCan_ = logLikelihood;
  double gLogLikelihood = 0.0;
  MPI::COMM_WORLD.Reduce(&logLikelihood,&gLogLikelihood,1,MPI::DOUBLE,MPI::SUM,0);

  return gLogLikelihood;
}

double
Mcmc::updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
    const double newTime)
{
  // Calculates an updated likelihood for an infection time move

  double logLikelihood = logLikelihood_;
  size_t counter = 0;
  Population<TestCovars>::PopulationIterator popj = pop_.asPop(j);

  // Sort out I1
  if (j == pop_.infecBegin() or newTime < pop_.infecBegin()->getI())
    {
      double oldTime = j->getI();
      pop_.moveInfectionTime(j, newTime);
      logLikelihood = calcLogLikelihood();
      pop_.moveInfectionTime(j, oldTime);
      return logLikelihood;
    }

  // Calculate logProduct
  // First instantaneous pressure on j
  Population<TestCovars>::InfectiveIterator infectee = pop_.infecBegin();
  Population<TestCovars>::InfectiveIterator stop;
  logProductCan_ = 0.0;
  size_t numInfecs = pop_.numInfected();

  // Temporarily move the infection time.
  double oldI = j->getI();
  pop_.moveInfectionTime(j,newTime);

  // Advance the iterator to our starting position
  advance(infectee, mpirank_);
  size_t pos = mpirank_;

  // First calculate the log product (happens on master node)
  if (infectee == pop_.infecBegin())
    {
      pos += mpiprocs_;
      if (pos < numInfecs) advance(infectee,mpiprocs_);
    }
  while (pos < numInfecs)
    {
      double tmp = instantPressureOn(infectee, infectee->getI());
      logProductCan_ += log(tmp);
      pos += mpiprocs_;
      advance(infectee, mpiprocs_);
    }

  logLikelihood -= logProduct_;
  logLikelihood += logProductCan_;

  // Move infection time back
  pop_.moveInfectionTime(j,oldI);

  // Integrated pressure on the movee
  if(mpirank_ == 0) {
      logLikelihood -= integPressureOn(popj,j->getI());
  }
  if(mpirank_ == mpiprocs_ - 1) {
      logLikelihood += integPressureOn(popj,newTime);
  }

  //Integral now
  Population<TestCovars>::PopulationIterator k = pop_.begin();
  advance(k, mpirank_);
  counter = mpirank_;
  while (counter < pop_.size())
    {
      if (k != popj)
        {
          double myBeta = beta(*j, *k);
          logLikelihood += myBeta * (min(j->getN(), k->getI()) - min(j->getI(),
              k->getI()));
          logLikelihood -= myBeta * (min(j->getN(), k->getI()) - min(newTime,
              k->getI()));
        }
      counter += mpiprocs_;
      advance(k, mpiprocs_);
    }

  logLikCan_ = logLikelihood;
  double gLogLikelihood = 0.0;
  MPI::COMM_WORLD.Reduce(&logLikelihood,&gLogLikelihood,1,MPI::DOUBLE,MPI::SUM,0);
  return gLogLikelihood;
}

bool
Mcmc::updateTrans()
{
  Parameters oldParams = params_;
  Random::Variates logvars;
  double* logvarsTmp = new double[params_.size()];
  double logPiCur;

  if (mpirank_ == 0)
    {
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

      for (size_t i = 0; i < params_.size(); ++i)
        logvarsTmp[i] = logvars(i);
    }

  MPI::COMM_WORLD.Bcast(logvarsTmp, params_.size(), MPI::DOUBLE, 0);

  params_(0) *= exp(logvarsTmp[0]);
  params_(1) *= exp(logvarsTmp[1]);
  params_(2) *= exp(logvarsTmp[2]);
  params_(3) *= exp(logvarsTmp[3]);

  delete[] logvarsTmp;

  double gLogLikCan = calcLogLikelihood();


  if (mpirank_ == 0)
    {
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
      if (log(random_->uniform()) < accept) {
        accept_ = true;
      }
      else {
        accept_ = false;
      }
    }

  MPI::COMM_WORLD.Bcast(&accept_, 1, MPI::BOOL, 0);

  if (accept_)
    {
      logLikelihood_ = logLikCan_;
      logProduct_ = logProductCan_;
      gLogLikelihood_ = gLogLikCan;
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
  if (mpirank_ == 0)
    {
      newI = random_->extreme(0.015, 0.8);
    }
  MPI::COMM_WORLD.Bcast(&newI, 1, MPI::DOUBLE, 0);

  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);
  newI = it->getN() - newI;

  double gLogLikCan = updateIlogLikelihood(it, newI);

  if (mpirank_ == 0)
    {
      double a = gLogLikCan - gLogLikelihood_;

      if (log(random_->uniform()) < a)
        {
          accept_ = true;
        }
      else {
          accept_ = false;
      }

    }

  MPI::COMM_WORLD.Bcast(&accept_, 1, MPI::BOOL, 0);

  if (accept_)
    {
      // Update the infection
      pop_.moveInfectionTime(it, newI);
      logLikelihood_ = logLikCan_;
      gLogLikelihood_ = gLogLikCan;
      logProduct_ = logProductCan_;
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
      acceptance["transParms"] = 0.0;
      acceptance["I"] = 0.0;
    }

  for (size_t k = 0; k < numIterations; ++k)
    {
      if (mpirank_ == 0 && k % 50 == 0)
        cout << "Iteration " << k << endl;

      acceptance["transParms"] += updateTrans();

      for (size_t infec = 0; infec < 20; ++infec)
        {
          if (mpirank_ == 0)
            toMove = random_->integer(pop_.numInfected());
          MPI::COMM_WORLD.Bcast(&toMove, 1, MPI::INT, 0);
          acceptance["I"] += updateI(toMove);
        }

      // Update the adaptive mcmc

      if (mpirank_ == 0)
        {
          logTransCovar_->sample();
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

//void
//Mcmc::moveProdCache(const string id, const size_t fromIndex, const size_t toIndex)
//{
//  int oldProcess = fromIndex % mpiprocs_;
//  int newProcess = toIndex % mpiprocs_;
//
//  if (oldProcess == newProcess) return;
//
//  if(mpirank_ == oldProcess) {
//      map<string,double>::iterator itId = productCacheTmp_.find(id);
//      if (itId == productCacheTmp_.end()) throw logic_error("Can't find id in product cache");
//      double idPressure = itId->second;
//      MPI::COMM_WORLD.Send(&idPressure,1,MPI::DOUBLE,newProcess,MPIPRODCACHE);
//      productCacheTmp_.erase(id);
//  }
//  else if (mpirank_ == newProcess) {
//      double idPressure = 0.0;
//      MPI::COMM_WORLD.Recv(&idPressure,1,MPI::DOUBLE,oldProcess,MPIPRODCACHE);
//      pair<map<string,double>::iterator,bool> rv = productCacheTmp_.insert(make_pair(id,idPressure));
//      if(rv.second != true) throw logic_error("Duplicate insert in destination product cache");
//  }
//}

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

//void
//Mcmc::dumpProdCache()
//{
//  for(size_t proc=0; proc<mpiprocs_; ++proc) {
//      MPI::COMM_WORLD.Barrier();
//      if(mpirank_ == proc) {
//      cout << "======================RANK " << proc << "====================" << endl;
//      map<string, double>::const_iterator it = productCache_.begin();
//      cout << "ID \t \t Cache \t \t TmpCache \t \t Difference\n" << endl;;
//      MPI::COMM_WORLD.Barrier();
//      while (it != productCache_.end())
//        {
//          cout << it->first << ":\t" << it->second << "\t"
//              << productCacheTmp_[it->first] << "\t" << productCacheTmp_[it->first]
//                                                                         - it->second << endl;
//          MPI::COMM_WORLD.Barrier();
//          ++it;
//        }
//      cout << "==============================================================\n" << endl;
//      }
//      MPI::COMM_WORLD.Barrier();
//  }
//}
