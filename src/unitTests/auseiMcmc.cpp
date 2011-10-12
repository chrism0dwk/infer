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

  void dumpData(Population<TestCovars>* popn)
  {
    for(Population<TestCovars>::PopulationIterator it = popn->begin();
        it != popn->end();
        it++)
      {
        const TestCovars& covars(it->getCovariates());
        cout << it->getId() << "\t"
             << covars.x << "\t"
             << covars.y << "\t"
             << covars.horses << "\t"
             << covars.area << "\n";
      }
  }



int main(int argc, char* argv[])
{
  // Tests out class Mcmc


  try {
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
  myPopulation->setObsTime(120.0);


  delete popDataImporter;
  delete epiDataImporter;

  cout << "Population size: " << myPopulation->size() << endl;
  cout << "Num infectives: " << myPopulation->numInfected() << endl;

  Parameters txparams(9);
  txparams(0) = Parameter(3e-4,GammaPrior(1,1),"epsilon");
  txparams(1) = Parameter(0.0019,GammaPrior(1,1),"gamma_1");
  txparams(2) = Parameter(0.01,GammaPrior(1,1),"gamma_2");
  txparams(3) = Parameter(0.1,GammaPrior(1,1),"xi");
  txparams(4) = Parameter(0.18,GammaPrior(1,1),"zeta");
  txparams(5) = Parameter(0.1,GammaPrior(1,1),"delta");
  txparams(6) = Parameter(0.13,GammaPrior(1,1),"alpha");
  txparams(7) = Parameter(7.0,GammaPrior(14.0,1),"a");
  txparams(8) = Parameter(0.4,GammaPrior(1,1),"b");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.01,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,2);

//  UpdateBlock updates;
//  for(size_t i=0; i<7; i++) updates.add(txparams(i));
//  AdaptiveMultiLogMRW* tx = myMcmc->newAdaptiveMultiLogMRW("allparams",updates, 1000);

  cout << "Adding updaters" << endl;
  myMcmc->newSingleSiteLogMRW(txparams(0),1.0);
  myMcmc->newSingleSiteLogMRW(txparams(1),0.03);
  myMcmc->newSingleSiteLogMRW(txparams(3),1.0);
  myMcmc->newSingleSiteLogMRW(txparams(4),0.5);
  myMcmc->newSingleSiteLogMRW(txparams(5),0.1);
  myMcmc->newWithinFarmBetaLogMRW(txparams(8),0.143,0.01);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/storage/stsiab/ausei/output/ausei_withinSIR6.parms";
  occFn << "/storage/stsiab/ausei/output/ausei_withinSIR6.occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[3]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  delete myMcmc;
  delete writer;
  delete myPopulation;

  }
  catch (std::exception& e) {
      cerr << "Exception occurred: " << e.what() << endl;
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}
