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
  txparams(3) = Parameter(5e-6,GammaPrior(1,1),"epsilon");
  txparams(4) = Parameter(0.18,GammaPrior(1,1),"xi_p");
  txparams(5) = Parameter(0.13,GammaPrior(1,1),"xi_s");
  txparams(6) = Parameter(0.72,GammaPrior(1,1),"psi_c");
  txparams(7) = Parameter(1.4,GammaPrior(1,1),"psi_p");
  txparams(8) = Parameter(0.018,GammaPrior(1,1),"psi_s");
  txparams(9) = Parameter(0.14,GammaPrior(1,1),"zeta_p");
  txparams(10) = Parameter(1.7,GammaPrior(1,1),"zeta_s");
  txparams(11) = Parameter(0.46,GammaPrior(1,1),"phi_c");
  txparams(12) = Parameter(0.5,GammaPrior(1,1),"phi_p");
  txparams(13) = Parameter(0.5,GammaPrior(1,1),"phi_s");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,1);

  ParameterView txInfec;
//  for(size_t i=0; i<txparams.size(); ++i)
  //txInfec.push_back(&txparams[0]);
  //txInfec.push_back(&txparams[1]);
  txInfec.push_back(&txparams[4]);
  txInfec.push_back(&txparams[5]);
  AdaptiveMultiLogMRW* tx = myMcmc->newAdaptiveMultiLogMRW("txInfec",txInfec, 1000);
  //tx->setCovariance(speciesCovar);

  ParameterView txSuscep;
  //txSuscep.push_back(&txparams[0]);
  //txSuscep.push_back(&txparams[1]);
  txSuscep.push_back(&txparams[9]);
  txSuscep.push_back(&txparams[10]);
  tx = myMcmc->newAdaptiveMultiLogMRW("txSuscep",txSuscep, 1000);
  tx->setCovariance(speciesCovar);

  ParameterView txDelta;
  txDelta.push_back(&txparams[0]);
  txDelta.push_back(&txparams[1]);
  txDelta.push_back(&txparams[2]);
  txDelta.push_back(&txparams[3]);
  tx = myMcmc->newAdaptiveMultiLogMRW("txDistance",txDelta, 1000);

  ParameterView txPsi;
  txPsi.push_back(&txparams[6]);
  txPsi.push_back(&txparams[7]);
  txPsi.push_back(&txparams[8]);
  myMcmc->newAdaptiveMultiLogMRW("txPsi",txPsi, 1000);

  ParameterView txPhi;
  txPhi.push_back(&txparams[11]);
  txPhi.push_back(&txparams[12]);
  txPhi.push_back(&txparams[13]);
  myMcmc->newAdaptiveMultiLogMRW("txPhi",txPhi, 1000);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/Users/stsiab/Documents/InFER/FMD2001/output/fmdTestPowMCK.p" << comm.size() << ".parms";
  occFn << "/Users/stsiab/Documents/InFER/FMD2001/output/fmdTestPowMCK.p" << comm.size() << ".occ";

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
