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
#include <queue>

//#ifdef __LINUX__
//#include <acml_mv.h>
//#define log fastlog
//#define exp fastexp
//#endif

#include "Mcmc.hpp"

using namespace EpiRisk;

// Constants
const double a = 7.0;
const double b = 0.5;
const double tuneI = 2.0;
const double numRUpdates = 100;
const double tuneGamma = 0.001;
const double ncProp = 0.5;

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

//inline
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
  pop_(population), txparams_(transParams), dxparams_(detectParams), integPressTime_(1.0),meanParams_(transParams),dicUpdates_(0),postMeanDev_(0.0)
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
  if (mpirank_ == 0) std::cout << "Log likelihood starts at " << logLikelihood_.global << std::endl;

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
  SingleSiteLogMRW* update = new SingleSiteLogMRW(param.getTag(), param, tuning, *random_, logLikelihood_, this);
  updateStack_.push_back(update);
}


//! Pushes an updater onto the MCMC stack
AdaptiveMultiLogMRW*
Mcmc::newAdaptiveMultiLogMRW(const string name, UpdateBlock& updateGroup, size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<LogTransform>::CovMatrix initCov(updateGroup.size());
  for(size_t i=0;i<updateGroup.size();++i)
    for(size_t j=0;j<=i;++j)
      initCov(i,j) = i == j ? 0.1 : 0.0;


  AdaptiveMultiLogMRW* update = new AdaptiveMultiLogMRW(name,updateGroup,burnin,*random_,logLikelihood_,this);
  updateStack_.push_back(update);

  return update;
}

//! Pushes an updater onto the MCMC stack
AdaptiveMultiMRW*
Mcmc::newAdaptiveMultiMRW(const string name, UpdateBlock& updateGroup, size_t burnin)
{
  // Create starting covariance matrix
  EmpCovar<Identity>::CovMatrix initCov(updateGroup.size());
  for(size_t i=0;i<updateGroup.size();++i)
    for(size_t j=0;j<=i;++j)
      initCov(i,j) = i == j ? 0.1 : 0.0;


  AdaptiveMultiMRW* update = new AdaptiveMultiMRW(name,updateGroup,burnin,*random_,logLikelihood_,this);
  updateStack_.push_back(update);

  return update;
}


//! Pushes an updater onto the MCMC stack
WithinFarmBetaLogMRW*
Mcmc::newWithinFarmBetaLogMRW(Parameter& param, const double gamma, const double tuning)
{
  WithinFarmBetaLogMRW* update = new WithinFarmBetaLogMRW(param,gamma,pop_,tuning,*random_,logLikelihood_,this);
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
Mcmc::h(const Population<TestCovars>::Individual& i, const double time) const
{
  if (time <= i.getI()) return 0.0;
  else return i.getCovariates().epi->numInfecAt(time - i.getI());
}

inline
double
Mcmc::H(const Population<TestCovars>::Individual& i, const double time) const
{
  if (time <= i.getI()) return 0.0;
  else return i.getCovariates().epi->integNumInfecAt(time - i.getI());
}


inline
double
Mcmc::infectivity(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const
{
  double infectivity = i.getCovariates().horses; //+ txparams_(3);// * i.getCovariates().area;
  return infectivity;
}

inline
double
Mcmc::susceptibility(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const
{
  double susceptibility = j.getCovariates().horses; //+ txparams_(4) * j.getCovariates().area;
  return susceptibility;
}

inline
double
Mcmc::beta(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x, i.getCovariates().y,
      j.getCovariates().x, j.getCovariates().y);
  if (distance <= 100.0)
    {
      return txparams_(1) /* infectivity(i,j) */ * susceptibility(i,j) * txparams_(5) / (txparams_(5)*txparams_(5) + distance*distance);
    }
  else
    return 0.0;
}

inline
double
Mcmc::betastar(const Population<TestCovars>::Individual& i, const Population<
    TestCovars>::Individual& j) const
{

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
          if (i->getN() >= Ij)
            {
              sumPressure += beta(*i, *j) * h(*i, Ij);
            }
          else if (i->getR() >= Ij)
            {
              sumPressure += betastar(*i, *j);
            }
        }
      ++i;
    }
  sumPressure += txparams_(0);

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
      //double hfunc = (H(*i,min(i->getN(), Ij)) - H(*i,min(i->getI(), Ij)));
      double hfunc = H(*i,Ij);// - H(*i, min(i->getI(), Ij));
      integPressure += beta(*i, *j) * hfunc;

      // Notified -> Susceptible pressure
      //integPressure += betastar(*i, *j) * (min(i->getR(), Ij) - min(i->getN(),
      //    Ij));
    }

  integPressure += txparams_(0) * (min(Ij, pop_.getObsTime()) - I1);

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
      double tmp = integPressureOn(k, k->getI());
      if ( tmp > 0.0 ) {
          logLikelihood.local = NEGINF;
          break;
      }
      else logLikelihood.local += tmp;
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
  for(it = processInfectives_.begin();
      it != processInfectives_.end();
      it++)
    {
      meanInfecTimes_.insert(make_pair((*it)->getId(),0.0));
    }
  dicUpdates_ = 0;
}

void
Mcmc::updateDIC()
{
  // Updates DIC

  dicUpdates_++;

  // Mean posterior deviance
  double delta = -2*getLogLikelihood() - postMeanDev_;
  postMeanDev_ += delta / dicUpdates_;

  // Mean posterior parameters
  for(size_t i=0; i<txparams_.size(); i++) {
      delta = txparams_(i) - meanParams_(i);
      meanParams_(i) += delta / dicUpdates_;
  }

  // Mean infection times
  map<string,double>::iterator it;
  for(it = meanInfecTimes_.begin();
      it != meanInfecTimes_.end();
      it++)
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
  map<string,double> oldInfecTimes;
  for(ProcessInfectives::iterator it = processInfectives_.begin();
      it != processInfectives_.end();
      it++)
    {
      Population<TestCovars>::InfectiveIterator i = *it;
      oldInfecTimes.insert(make_pair(i->getId(),i->getI()));
      pop_.moveInfectionTime(i,meanInfecTimes_[i->getId()]);
    }

  Likelihood myLikelihood;
  calcLogLikelihood(myLikelihood);

  // Assemble DIC
  DIC rv;
  rv.Dbar = postMeanDev_;
  rv.Dhat = -2*myLikelihood.global;
  rv.pD = rv.Dbar - rv.Dhat;
  rv.DIC = rv.Dbar - rv.pD;

  // Undo changes
  txparams_ = oldParams;
  for(ProcessInfectives::iterator it = processInfectives_.begin();
        it != processInfectives_.end();
        it++)
      {
        pop_.moveInfectionTime(*it, oldInfecTimes[(*it)->getId()]);
      }

  return rv;
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
            myPressure -= beta(*j, **i) * h(*j,(*i)->getI());

          if (newTime <= (*i)->getI() && (*i)->getI() < j->getN())
            myPressure += beta(*j, **i) * h(*j,(*i)->getI());  // THIS AIN'T CORRECT!!!
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
          logLikelihood += myBeta * (H(*j,min(j->getN(), i->getI())) - H(*j,min(j->getI(), // NOT CORRECT!!!
              i->getI())));
          logLikelihood -= myBeta * (H(*j,min(j->getN(), i->getI())) - H(*j,min(newTime, // NOT CORRECT!!!
              i->getI())));
        }
    }

  updatedLogLik.local = logLikelihood;
  all_reduce(comm_, updatedLogLik.local, updatedLogLik.global, plus<double>());
}



void
Mcmc::updateRlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
    const double newTime, Likelihood& updatedLogLik)
{
  // Calculates an updated likelihood for an infection time move

  double logLikelihood = logLikelihood_.local;
  Population<TestCovars>::PopulationIterator popj = pop_.asPop(j);
  updatedLogLik.productCache.clear();

  // Product part of likelihood
  for (ProcessInfectives::const_iterator i = processInfectives_.begin(); i
      != processInfectives_.end(); ++i)
    {

      map<string, double>::const_iterator it = logLikelihood_.productCache.find((*i)->getId());
      if(it == logLikelihood_.productCache.end()) throw logic_error("Individual non-existent in product cache");

      double myPressure = it->second;

      logLikelihood -= log(myPressure);

      if (j->isNAt((*i)->getI())) {
        myPressure -= betastar(*j, **i);
      }
      if (j->getN() < (*i)->getI() && (*i)->getI() <= newTime) {
          double tmp = betastar(*j, **i);
          myPressure += tmp;
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
        continue;
      else
        {
          double myBetastar = betastar(*j, *i);
          logLikelihood += myBetastar * (min(j->getR(), i->getI()) - min(j->getN(),
              i->getI()));
          logLikelihood -= myBetastar * (min(newTime, i->getI()) - min(j->getN(),
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
Mcmc::updateR(const size_t index)
{
  Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
  advance(it, index);

  double newR = it->getN() + (it->getR() - it->getN()) * exp(random_->gaussian(0,tuneI));
  //double newI = it->getN() - random_->extreme(a, b); // Independence sampler

  Likelihood logLikCan;
  updateRlogLikelihood(it, newR, logLikCan);
  //all_reduce(comm_, logLikCan.local, logLikCan.global, plus<double> ());

  double piCan = logLikCan.global + log(gsl_ran_gamma_pdf(newR - it->getN(),txparams_(7),1.0/txparams_(8)));
  double piCur = logLikelihood_.global + log(gsl_ran_gamma_pdf(it->getR() - it->getN(),txparams_(7),1.0/txparams_(8)));

  double qRatio = log((newR - it->getN()) / (it->getR() - it->getN()));
  double accept = piCan - piCur + qRatio;

  if (log(random_->uniform()) < accept)
    {
      // Update the infection
      pop_.moveRemovalTime(it->getId(), newR);
      logLikelihood_ = logLikCan;
      return true;
    }
  else
    {
      return false;
    }
}


bool
Mcmc::updateB()
{
  // Updates b hyperparameter of prior on Ri - Ni

  const double lambda = 1.0;
  const double nu = 1.0;

  double sumNminusR = 0.0;

  for(Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
      it != pop_.infecEnd();
      it++) sumNminusR += it->getR() - it->getN();

  double a = pop_.numInfected()*txparams_(7) + lambda;
  double b = sumNminusR + nu;

  txparams_(8) = random_->gamma(a,b);

  return true;
}


bool
Mcmc::updateBpnc()
{
  // Partially non-centred update for b

  double oldB = txparams_(8);

  // Current prior value for b
  double logPiCur = log(txparams_(8).prior());

  // Propose gamma by logRW
  txparams_(8) = txparams_(8) * exp(random_->gaussian(0,tuneGamma));
  double logPiCan = log(txparams_(8).prior());

  // Mark infection times for non-centering with probability ncProp
  // add to conditional posterior (NC likelihood components cancel)
  typedef list< Population<TestCovars>::InfectiveIterator > UndoList;
  UndoList undoList;
  for(Population<TestCovars>::InfectiveIterator it = pop_.infecBegin();
      it != pop_.infecEnd();
      it++) {

      if(random_->uniform() < ncProp) undoList.push_front(it);
      else {
          logPiCur += log( gsl_ran_gamma_pdf(it->getR() - it->getN(), txparams_(7), 1/oldB) );
          logPiCan += log( gsl_ran_gamma_pdf(it->getR() - it->getN(), txparams_(7), 1/txparams_(8)) );
      }
  }

  // Non-center individuals who are marked
  for(UndoList::const_iterator it = undoList.begin();
      it != undoList.end();
      it++) {
      double ncR = ( ((*it)->getR() - (*it)->getN()) * oldB / txparams_(8) ) + (*it)->getN() ;
      pop_.moveRemovalTime((*it)->getId(),ncR);
  }

  // Calculate log likelihood
  Likelihood logLikCan;
  calcLogLikelihood(logLikCan);

  logPiCur += logLikelihood_.global;
  logPiCan += logLikCan.global;

  // q-ratio
  double qRatio = log( txparams_(8) / oldB );

  // Accept/reject
  if(log(random_->uniform()) < logPiCan - logPiCur + qRatio) {
      // Accept
      logLikelihood_ = logLikCan;
      return true;
  }
  else {
      // Reject the move, resetting the changes to the removal times
      while(!undoList.empty()) {
          double oldR = ( (undoList.front()->getR() - undoList.front()->getN()) * txparams_(8) / oldB ) + undoList.front()->getN() ;
          pop_.moveRemovalTime(undoList.front()->getId(),oldR);
          undoList.pop_front();
      }
      txparams_(8) = oldB;
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
      writer.open(txparams_);
      writer.write(pop_);
      writer.write(txparams_);
    }

  acceptance["R"] = 0.0;
  acceptance["gamma"] = 0.0;

  cout << "Running MCMC..." << flush;
  for (size_t k = 0; k < numIterations; ++k)
    {
      if (mpirank_ == 0 && k % 1 == 0)
        cout << "Iteration " << k << endl;

      if(k % 100 == 0) {
          DIC myDIC = getDIC();
          if (mpirank_ == 0) {
              cout << "=======DIC=======\n"
                   << "Dbar: " << myDIC.Dbar << "\n"
                   << "Dhat: " << myDIC.Dhat << "\n"
                   << "pD: " << myDIC.pD << "\n"
                   << "DIC: " << myDIC.DIC << endl;
          }
      }

      for(boost::ptr_list<McmcUpdate>::iterator it = updateStack_.begin();
          it != updateStack_.end();
          ++it) it->update();

//      for (size_t infec = 0; infec < numRUpdates; ++infec)
//        {
//          toMove = random_->integer(pop_.numInfected());
//          acceptance["R"] += updateR(toMove);
//        }
//
//      acceptance["gamma"] += updateBpnc();

      updateDIC();

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
          ++it)
        {
          cout << it->getTag() << ": " << it->getAcceptance() << "\n";
        }
      cout << "Recovery times: " << acceptance["R"] / (numIterations * numRUpdates) << "\n";
      cout << "b: " << acceptance["gamma"] / numIterations << "\n";
      cout << "============\n";
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
