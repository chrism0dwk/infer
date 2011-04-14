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

  if (argc != 5) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <covar matrix> <num iterations>" << endl;
      return EXIT_FAILURE;
  }

  typedef Population<TestCovars> MyPopulation;

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  myPopulation->setObsTime(241.0);

  delete popDataImporter;
  delete epiDataImporter;

  // Data covariance matrix
  EmpCovar<LogTransform>::CovMatrix speciesCovar;
  ifstream covMatrix;
  covMatrix.open(argv[3],ios::in);
  covMatrix >> speciesCovar;
  covMatrix.close();


  Parameters txparams(14);
  txparams(0) = Parameter(2e-6,GammaPrior(1,1),"gamma1");
  txparams(1) = Parameter(1.8e-7,GammaPrior(1,1),"gamma2");
  txparams(2) = Parameter(1,GammaPrior(1,1),"delta");
  txparams(3) = Parameter(1e-6,GammaPrior(1,1),"epsilon");
  txparams(4) = Parameter(0.18,GammaPrior(3,3),"xi_p");
  txparams(5) = Parameter(0.13,GammaPrior(1,1),"xi_s");
  txparams(6) = Parameter(1,BetaPrior(2,2),"psi_c");
  txparams(7) = Parameter(1,BetaPrior(2,2),"psi_p");
  txparams(8) = Parameter(1,BetaPrior(2,2),"psi_s");
  txparams(9) = Parameter(0.14,GammaPrior(3,3),"zeta_p");
  txparams(10) = Parameter(1.7,GammaPrior(1,1),"zeta_s");
  txparams(11) = Parameter(1,BetaPrior(2,2),"phi_c");
  txparams(12) = Parameter(1,BetaPrior(2,2),"phi_p");
  txparams(13) = Parameter(1,BetaPrior(2,2),"phi_s");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,1);

//  myMcmc->newSingleSiteLogMRW(txparams[0],0.2);
//  myMcmc->newSingleSiteLogMRW(txparams[1],0.4);
//  myMcmc->newSingleSiteLogMRW(txparams[2],0.2);
//  myMcmc->newSingleSiteLogMRW(txparams[3],0.1);
//  myMcmc->newSingleSiteLogMRW(txparams[4],3.0);
//  myMcmc->newSingleSiteLogMRW(txparams[5],0.4);
//  myMcmc->newSingleSiteLogMRW(txparams[9],3.0);
//  myMcmc->newSingleSiteLogMRW(txparams[10],0.4);



  UpdateBlock txInfec;
  InfSuscSN xi_p(txparams[0],txparams[4]);
  InfSuscSN xi_s(txparams[0],txparams[5]);
  txInfec.add(txparams[0]);
  txInfec.add(txparams[1]);
  txInfec.add(&xi_p);
  txInfec.add(&xi_s);
  AdaptiveMultiLogMRW* tx = myMcmc->newAdaptiveMultiLogMRW("txInfec",txInfec, 1000);
  //tx->setCovariance(speciesCovar);

  UpdateBlock txSuscep;
  //InfSuscSN zeta_p(txparams[0],txparams[9]);
  //InfSuscSN zeta_s(txparams[0],txparams[10]);
  txSuscep.add(txparams[0]);
  txSuscep.add(txparams[1]);
  txSuscep.add(txparams[9]);
  txSuscep.add(txparams[10]);
  tx = myMcmc->newAdaptiveMultiLogMRW("txSuscep",txSuscep, 1000);
  //tx->setCovariance(speciesCovar);

  UpdateBlock txDelta;
  txDelta.add(txparams[2]);
  txDelta.add(txparams[3]);
  tx = myMcmc->newAdaptiveMultiLogMRW("txDistance",txDelta, 1000);

  UpdateBlock txPsi;
  txPsi.add(txparams[6]);
  txPsi.add(txparams[7]);
  txPsi.add(txparams[8]);
  myMcmc->newAdaptiveMultiLogMRW("txPsi",txPsi, 1000);

  UpdateBlock txPhi;
  txPhi.add(txparams[11]);
  txPhi.add(txparams[12]);
  txPhi.add(txparams[13]);
  myMcmc->newAdaptiveMultiLogMRW("txPhi",txPhi, 1000);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/Users/stsiab/Documents/InFER/FMD2001/output/fmdTestPowSN.p" << comm.size() << ".parms";
  occFn << "/Users/stsiab/Documents/InFER/FMD2001/output/fmdTestPowSN.p" << comm.size() << ".occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[4]);
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
