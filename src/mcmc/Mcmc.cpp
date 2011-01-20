/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <cmath>
#include <boost/numeric/ublas/io.hpp>
#include <boost/mpi/datatype.hpp>
#include <boost/mpi/collectives.hpp>
#include <ctime>

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
const double tuneI = 0.8;
const double numIUpdates = 0;

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
gaussianTailPdf(const double x, const double mean, const double var)
{
  return gsl_ran_gaussian_tail_pdf(x-mean,-mean,sqrt(var));
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

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& transParams, Parameters& detectParams,
    const size_t randomSeed) :
  pop_(population), txparams_(transParams), dxparams_(detectParams), logLikelihood_(0), integPressTime_(1.0)
{

  // MPI setup
  int largc = 1;
  char** largv;

  mpirank_ = comm_.rank();
  mpiprocs_ = comm_.size();

  // Random number generation
  random_ = new Random(randomSeed);

  //AdMCMC covariance.
  EmpCovar<ExpTransform>::CovMatrix initCov(txparams_.size());
  for (size_t i = 0; i < txparams_.size(); ++i)
    {
      for (size_t j = 0; j < txparams_.size(); ++j)
        {
            initCov(i, j) = 0.0;
        }
    }

  initCov(0,0) = 0.01;
  initCov(1,1) = 0.01;
  initCov(2,2) = 0.01;
  initCov(3,3) = 0.01;
  initCov(4,4) = 0.01; //0.95; // Inf pigs
  initCov(5,5) = 0.01; //0.1; // Inf sheep
  initCov(6,6) = 0.01; //2.53; // Inf goats
  initCov(7,7) = 0.01; //154.4; // Inf deer
  initCov(8,8) = 0.01; //0.95; // Susc pigs
  initCov(9,9) = 0.01; //0.1; // Susc sheep
  initCov(10,10) = 0.01; //2.53; // Susc goats
  initCov(11,11) = 0.01; //154.4; // Susc deer

  logTransCovar_ = new EmpCovar<LogTransform> (txparams_, initCov);
  stdCov_ = new ublas::matrix<double>(txparams_.size(), txparams_.size());

  for (size_t i = 0; i < txparams_.size(); ++i)
    {
      for (size_t j = 0; j < txparams_.size(); ++j)
        {
          if (i == j)
            (*stdCov_)(i, j) = 0.01 / txparams_.size();
          else
            (*stdCov_)(i, j) = 0.0;
        }

    }

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
  if (distance <= 25.0)
    {
      double infectivity = i.getCovariates().cattle +
                           txparams_(4)*i.getCovariates().pigs +
                           txparams_(5)*i.getCovariates().sheep +
                           txparams_(6)*i.getCovariates().goats +
                           txparams_(7)*i.getCovariates().deer;
      double susceptibility = j.getCovariates().cattle +
                                 txparams_(8)*j.getCovariates().pigs +
                                 txparams_(9)*j.getCovariates().sheep +
                                 txparams_(10)*j.getCovariates().goats +
                                 txparams_(11)*j.getCovariates().deer;

      return txparams_(0) * infectivity * susceptibility * txparams_(2) / (txparams_(2)*txparams_(2) + distance*distance);
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
  if (distance <= 25.0) {
      double infectivity = i.getCovariates().cattle +
                           txparams_(4)*i.getCovariates().pigs +
                           txparams_(5)*i.getCovariates().sheep +
                           txparams_(6)*i.getCovariates().goats +
                           txparams_(7)*i.getCovariates().deer;
      double susceptibility = j.getCovariates().cattle +
                                 txparams_(8)*j.getCovariates().pigs +
                                 txparams_(9)*j.getCovariates().sheep +
                                 txparams_(10)*j.getCovariates().goats +
                                 txparams_(11)*j.getCovariates().deer;

      return txparams_(1) * infectivity * susceptibility * txparams_(2) / (txparams_(2)*txparams_(2) + distance*distance);
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
      integPressure += beta(*i, *j) * (min(i->getN(), Ij) - min(i->getI(), Ij));

      // Notified -> Susceptible pressure
      integPressure += betastar(*i, *j) * (min(i->getR(), Ij) - min(i->getN(),
          Ij));
    }

  integPressure += txparams_(3) * (min(Ij, pop_.getObsTime()) - I1);

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
  size_t pos;
  Population<TestCovars>::PopulationIterator k;
  for (pos = mpirank_, k = pop_.begin() + mpirank_;
       pos < pop_.size();
       pos += mpiprocs_, k += mpiprocs_)
    {
      logLikelihood += integPressureOn(k, k->getI());
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
  size_t pos;
  Population<TestCovars>::PopulationIterator i;
  for (pos = mpirank_, i = pop_.begin() + mpirank_;
       pos < pop_.size();
       pos += mpiprocs_, i += mpiprocs_)
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
    }

  return logLikelihood;
}


//BlockUpdate&
//Mcmc::createBlockUpdate()
//{
//  updates_.push_back(BlockUpdate);
//  return updates_.back();
//}


bool
Mcmc::updateTrans()
{
  Parameters oldParams = txparams_;
  Random::Variates logvars;
  double logPiCur;
  double gLogLikCan;

  logPiCur = gLogLikelihood_;
  for(size_t p=0; p < txparams_.size(); ++p)
      logPiCur += log(txparams_(p).prior());

  if (random_->uniform() < 0.95)
    {
      try
        {
          logvars = random_->mvgauss(logTransCovar_->getCovariance() * 2.38
				     * 2.38 / txparams_.size());
        }
      catch (cholesky_error& e)
        {
          logvars = random_->mvgauss(*stdCov_);
        }
    }
  else
    logvars = random_->mvgauss(*stdCov_);

  for(size_t p=0; p<txparams_.size(); ++p)
    txparams_(p) *= exp(logvars[p]);

  double logLikCan = calcLogLikelihood();
  all_reduce(comm_, logLikCan, gLogLikCan, plus<double> ());

  double logPiCan = gLogLikCan;
  for(size_t p=0; p<txparams_.size(); ++p)
    logPiCan += log(txparams_(p).prior());

  double qRatio = 0.0;
  for(size_t p=0; p<txparams_.size(); ++p)
    qRatio += log(txparams_(p) / oldParams(p));

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
      for(size_t p=0; p<txparams_.size();++p)
        txparams_(p) = oldParams(p);
      return false;
    }

}


bool
Mcmc::updateATrans(const size_t p, const double tune)
{
  if(p >= txparams_.size()) throw range_error("Specified parameter out of range");

  Parameters oldParams = txparams_;
  double logPiCur;
  double gLogLikCan;

  logPiCur = gLogLikelihood_;
  logPiCur += log(txparams_(p).prior());

  txparams_(p) *= exp(random_->gaussian(0,tune));

  double logLikCan = calcLogLikelihood();
  all_reduce(comm_, logLikCan, gLogLikCan, plus<double> ());

  double logPiCan = gLogLikCan;
  logPiCan += log(txparams_(p).prior());

  double qRatio = 0.0;
  qRatio += log(txparams_(p) / oldParams(p));

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
      txparams_(p) = oldParams(p);
      return false;
    }

}



bool
Mcmc::updateI(const size_t index)
{
  double gLogLikCan;

  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

  double newI = it->getN() - (it->getN() - it->getI()) * exp(random_->gaussian(0,tuneI));
  //double newI = it->getN() - random_->extreme(a, b); // Independence sampler

  double logLikCan = updateIlogLikelihood(it, newI);
  all_reduce(comm_, logLikCan, gLogLikCan, plus<double> ());

  double piCan = gLogLikCan + log(extremepdf(it->getN() - newI,a,b));
  double piCur = gLogLikelihood_ + log(extremepdf(it->getN() - it->getI(),a,b));

  double qRatio = log((it->getN() - newI) / (it->getN() - it->getI()));
  double accept = piCan - piCur + qRatio;

  if (log(random_->uniform()) < accept)
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

bool
Mcmc::addI()
{
  size_t numSusceptible = pop_.numSusceptible();

  Population<TestCovars>::InfectiveIterator it = pop_.infecEnd();
  advance(it,random_->integer(numSusceptible));

  double inProp = random_->gaussianTail(-(1/b), 1/(a*b*b));
  double newI = pop_.getObsTime() - inProp;

  double logPiCur = gLogLikelihood_;

  double logLikCan = updateIlogLikelihood(it,newI);
  double glogLikCan;
  all_reduce(comm_,logLikCan,glogLikCan,plus<double>());

  double logPiCan = glogLikCan + log(extremepdf(inProp,a,b));

  double qRatio = log( 1.0 / (occultList_.size() + 1) / (1.0 / numSusceptible * gaussianTailPdf(inProp,-1/b,1/(a*b*b))));

  double accept = logPiCan - logPiCur + qRatio;
  // Perform accept/reject step.

  if(log(random_->uniform()) < accept)
    {
      pop_.moveInfectionTime(it,newI);
      logLikelihood_ = logLikCan;
      gLogLikelihood_ = glogLikCan;
      productCache_ = productCacheTmp_;
      return true;
    }
  else
    {
      return false;
    }
}

bool
Mcmc::deleteI()
{
  size_t numSusceptible = pop_.numSusceptible();

  ProcessInfectives::const_iterator idx = occultList_.begin();
  advance(idx,random_->integer(occultList_.size()));

  Population<TestCovars>::InfectiveIterator it = *idx;

  double logPiCur = gLogLikelihood_ + log(extremepdf(it->getN() - it->getI(),a,b));

  double logLikCan = updateIlogLikelihood(it,POSINF);
  double glogLikCan;
  all_reduce(comm_,logLikCan, glogLikCan, plus<double>());

  double logPiCan = glogLikCan;

  double qRatio = log( 1.0/(numSusceptible + 1) * gaussianTailPdf(it->getN() - it->getI(),-1/b,1/(a*b*b))  / 1.0/occultList_.size());
  // Perform accept/reject step.

  double accept = logPiCan - logPiCur + qRatio;

  if(log(random_->uniform()) < accept)
    {
      pop_.moveInfectionTime(it,POSINF);
      logLikelihood_ = logLikCan;
      gLogLikelihood_ = glogLikCan;
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

  writer.write(pop_);
  writer.write(txparams_);

//  acceptance["alpha"] = 0.0;
//  acceptance["alphastar"] = 0.0;
//  acceptance["delta"] = 0.0;
//  acceptance["epsilon"] = 0.0;
  acceptance["transParms"] = 0.0;
  acceptance["I"] = 0.0;

  for (size_t k = 0; k < numIterations; ++k)
    {
      if (mpirank_ == 0 && k % 1 == 0)
        cout << "Iteration " << k << endl;

      acceptance["transParms"] += updateTrans();

      //if (k % 10 == 0) loadBalance();

      for (size_t infec = 0; infec < numIUpdates; ++infec)
        {
          toMove = random_->integer(pop_.numInfected());
          acceptance["I"] += updateI(toMove);
        }

      if(mpirank_ == 0) cout << "gLogLikelihood: " << gLogLikelihood_ << endl;

      // Update the adaptive mcmc
      logTransCovar_->sample();
      if (mpirank_ == 0)
        {
          writer.write(pop_);
          writer.write(txparams_);
        }
    }

  if (mpirank_ == 0)
    {
      cout << "\n";
      cout << "Covariance: " << logTransCovar_->getCovariance() << endl;
      logTransCovar_->printInnerds();

      writer.close();
      acceptance["transParms"] /= (double)numIterations;
      acceptance["I"] /= (numIterations * numIUpdates);
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

void
Mcmc::loadBalance()
{
  std::vector<double> times;
  multimap<double,size_t> processors; // ordered by ascending speed
  mpi::all_gather(comm_,integPressTime_,times);

  if (mpirank_ == 0) {
      cout << "Times: ";
      for(size_t i=0; i<times.size(); ++i) cout << times[i] << " ";
      cout << endl;
  }


  double timesSum = 0.0;
  for(size_t i=0;i<times.size();++i) {
      times[i] = 1/times[i];
      processors.insert(make_pair(times[i],i));
      timesSum += times[i];
  }

  int leftover = 0;
  for(size_t i=0;i<times.size();++i) {
      times[i] = times[i] / timesSum * pop_.size();
      elements_[i] = round(times[i]);
      leftover += elements_[i];
  }

  leftover = pop_.size() - leftover;

  if(leftover < 0) // Remove elements from slowest processors
    {
      map<double,size_t>::const_iterator it = processors.begin();
      while(leftover !=0) {
          elements_[it->second] -= 1;
          leftover += 1;
          it++;
      }
    }
  else if (leftover > 0) // Add elements to fastest processors
    {
      map<double,size_t>::const_iterator it = processors.end();
      while(leftover != 0) {
          it--;
          elements_[it->second] += 1;
          leftover -= 1;
      }
    }

  if (mpirank_ == 0) {
      cout << "Elements: ";
      for(size_t i=0; i<elements_.size(); ++i) cout << elements_[i] << " ";
      cout << endl;
  }


}
