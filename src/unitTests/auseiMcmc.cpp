/*
 * testMcmc.cpp
 *
 *  Created on: Oct 15, 2010
 *      Author: stsiab
 */

#include <iostream>
#include <gsl/gsl_randist.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <sstream>

#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"


using namespace EpiRisk;

  class GammaPrior : public Prior
  {
    double shape_;
    double rate_;
  public:
    GammaPrior(const double shape, const double rate)
    {
      shape_ = shape;
      rate_ = rate;
    }
    double
    operator()(const double x)
    {
      return gsl_ran_gamma_pdf(x,shape_,1/rate_);
    }
    Prior*
    create() const
    {
      return new GammaPrior(shape_,rate_);
    }
    Prior*
    clone() const
    {
      return new GammaPrior(*this);
    }
  };

  class BetaPrior : public Prior
  {
    double a_;
    double b_;
  public:
    BetaPrior(const double a, const double b) : a_(a),b_(b) {};
    double operator()(const double x)
    {
      return gsl_ran_beta_pdf(x,a_,b_);
    }
    Prior*
    create() const
    {
      return new BetaPrior(a_,b_);
    }
    Prior*
    clone() const
    {
      return new BetaPrior(*this);
    }
  };


  class InfSuscSN : public StochasticNode
  {
    Parameter* A_;
  public:
    InfSuscSN(Parameter& A, Parameter& B) : A_(&A), StochasticNode(B)
    {
    }
    InfSuscSN*
    clone()
    {
      return new InfSuscSN(*this);
    }
    double getValue() const
    {
      return *A_ * *param_;
    }
    void setValue(const double x)
    {
      *param_ = x / *A_;
    }
  };



int main(int argc, char* argv[])
{
  // Tests out class Mcmc

  mpi::environment env(argc,argv);
  mpi::communicator comm;

  if (argc != 4) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <num iterations>" << endl;
      return EXIT_FAILURE;
  }

  typedef Population<TestCovars> MyPopulation;

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  myPopulation->setObsTime(145.0);

  delete popDataImporter;
  delete epiDataImporter;

  Parameters txparams(7);
  txparams(0) = Parameter(2e-6,GammaPrior(1,1),"epsilon");
  txparams(1) = Parameter(0.01,GammaPrior(1,1),"gamma_1");
  txparams(2) = Parameter(0.01,GammaPrior(1,1),"gamma_2");
  txparams(3) = Parameter(0.1,GammaPrior(1,1),"xi");
  txparams(4) = Parameter(0.18,GammaPrior(1,1),"zeta");
  txparams(5) = Parameter(1.13,GammaPrior(1,1),"delta");
  txparams(6) = Parameter(0.13,GammaPrior(1,1),"alpha");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,1);

//  UpdateBlock updates;
//  for(size_t i=0; i<7; i++) updates.add(txparams(i));
//  AdaptiveMultiLogMRW* tx = myMcmc->newAdaptiveMultiLogMRW("allparams",updates, 1000);

  myMcmc->newSingleSiteLogMRW(txparams(0),8.0);
  myMcmc->newSingleSiteLogMRW(txparams(1),0.8);
  myMcmc->newSingleSiteLogMRW(txparams(2),0.4);
  //myMcmc->newSingleSiteLogMRW(txparams(3),1000.0);
  //myMcmc->newSingleSiteLogMRW(txparams(4),0.1);
  myMcmc->newSingleSiteLogMRW(txparams(5),0.7);
  //myMcmc->newSingleSiteLogMRW(txparams(6),0.4);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/Users/stsiab/Documents/Australia/Simon/output/ausei.parms";
  occFn << "/Users/stsiab/Documents/Australia/Simon/output/ausei.occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[3]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  if(comm.rank() == 0) {
      cout << "Infection acceptance: " << acceptance["I"] << endl;
  }

  delete myMcmc;
  delete writer;
  delete myPopulation;

  return EXIT_SUCCESS;

}
