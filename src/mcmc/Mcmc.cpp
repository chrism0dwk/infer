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
const double numIUpdates = 200;

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
  pop_(population), txparams_(transParams), dxparams_(detectParams), integPressTime_(1.0)
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
  all_reduce(comm_, logLikelihood_.local, logLikelihood_.global, plus<double> ());
}

Mcmc::~Mcmc()
{
  delete random_;
}


//! Pushes an updater onto the MCMC stack
AdaptiveMultiLogMRW*
Mcmc::newAdaptiveMultiLogMRW(const string name, ParameterView& updateGroup)
{
  // Create starting covariance matrix
  EmpCovar<LogTransform>::CovMatrix initCov(updateGroup.size());
  for(size_t i=0;i<updateGroup.size();++i)
    for(size_t j=0;j<=i;++j)
      initCov(i,j) = i == j ? 0.1 : 0.0;


  AdaptiveMultiLogMRW* update = new AdaptiveMultiLogMRW(name,updateGroup,*random_,logLikelihood_,this);
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
  for (pos = mpirank_, k = pop_.begin() + mpirank_;
       pos < pop_.size();
       pos += mpiprocs_, k += mpiprocs_)
    {
      logLikelihood.local += integPressureOn(k, k->getI());
    }

  all_reduce(comm_, logLikelihood.local, logLikelihood.global, plus<double> ());
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
      double myPressure = logLikelihood_.productCache.find((*i)->getId())->second;
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

      updatedLogLik.productCache.insert(make_pair((*i)->getId(), myPressure));
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

  updatedLogLik.local = logLikelihood;
  all_reduce(comm_, updatedLogLik.local, updatedLogLik.global, plus<double>());
}


bool
Mcmc::updateI(const size_t index)
{
  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

  double newI = it->getN() - (it->getN() - it->getI()) * exp(random_->gaussian(0,tuneI));
  //double newI = it->getN() - random_->extreme(a, b); // Independence sampler

  Likelihood logLikCan;
  updateIlogLikelihood(it, newI, logLikCan);
  all_reduce(comm_, logLikCan.local, logLikCan.global, plus<double> ());

  double piCan = logLikCan.global + log(extremepdf(it->getN() - newI,a,b));
  double piCur = logLikelihood_.global + log(extremepdf(it->getN() - it->getI(),a,b));

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
  size_t numSusceptible = pop_.numSusceptible();

  Population<TestCovars>::InfectiveIterator it = pop_.infecEnd();
  advance(it,random_->integer(numSusceptible));

  double inProp = random_->gaussianTail(-(1/b), 1/(a*b*b));
  double newI = pop_.getObsTime() - inProp;

  double logPiCur = logLikelihood_.global;

  Likelihood logLikCan;
  updateIlogLikelihood(it,newI, logLikCan);
  double glogLikCan;
  all_reduce(comm_,logLikCan.local,logLikCan.global,plus<double>());

  double logPiCan = glogLikCan + log(extremepdf(inProp,a,b));

  double qRatio = log( 1.0 / (occultList_.size() + 1) / (1.0 / numSusceptible * gaussianTailPdf(inProp,-1/b,1/(a*b*b))));

  double accept = logPiCan - logPiCur + qRatio;
  // Perform accept/reject step.

  if(log(random_->uniform()) < accept)
    {
      pop_.moveInfectionTime(it,newI);
      logLikelihood_ = logLikCan;
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

  double logPiCur = logLikelihood_.global + log(extremepdf(it->getN() - it->getI(),a,b));

  Likelihood logLikCan;
  updateIlogLikelihood(it,POSINF, logLikCan);
  double glogLikCan;
  all_reduce(comm_,logLikCan.local, logLikCan.global, plus<double>());

  double logPiCan = glogLikCan;

  double qRatio = log( 1.0/(numSusceptible + 1) * gaussianTailPdf(it->getN() - it->getI(),-1/b,1/(a*b*b))  / 1.0/occultList_.size());
  // Perform accept/reject step.

  double accept = logPiCan - logPiCur + qRatio;

  if(log(random_->uniform()) < accept)
    {
      pop_.moveInfectionTime(it,POSINF);
      logLikelihood_ = logLikCan;
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
      writer.write(pop_);
      writer.write(txparams_);
    }

  acceptance["I"] = 0.0;

  for (size_t k = 0; k < numIterations; ++k)
    {
      if (mpirank_ == 0 && k % 1 == 0)
        cout << "Iteration " << k << endl;

      for(boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin();
          it != updateStack_.end();
          ++it) it->update();

      for (size_t infec = 0; infec < numIUpdates; ++infec)
        {
          toMove = random_->integer(pop_.numInfected());
          acceptance["I"] += updateI(toMove);
        }

      if(mpirank_ == 0) cout << "gLogLikelihood: " << logLikelihood_.global << endl;

      // Update the adaptive mcmc
      if (mpirank_ == 0)
        {
          writer.write(pop_);
          writer.write(txparams_);
        }
    }

  if (mpirank_ == 0)
    {
      cout << "\n";
//      logTransCovar_->printInnerds();

      writer.close();
      cout << "Acceptances:\n";
      cout << "============\n";
      for(boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin();
          it != updateStack_.end();
          ++it) cout << it->getTag() << ": " << it->getAcceptance() << "\n";
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
          map<string, double>::const_iterator it = logLikelihood_.productCache.begin();
          cout << "ID \t \t Cache \t \t TmpCache \t \t Difference\n" << endl;
          ;
          MPI::COMM_WORLD.Barrier();
          while (it != logLikelihood_.productCache.end())
            {
              cout << it->first << ":\t" << it->second << "\t"
                  << endl;
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
