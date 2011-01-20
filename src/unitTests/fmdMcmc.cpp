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

namespace mpi = boost::mpi;

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
  myPopulation->setObsTime(241.0);

  delete popDataImporter;
  delete epiDataImporter;

  Parameters txparams(12);
  txparams(0) = Parameter(0.005,GammaPrior(1,1));
  txparams(1) = Parameter(0.004,GammaPrior(1,1));
  txparams(2) = Parameter(1.2,GammaPrior(1,1));
  txparams(3) = Parameter(2e-6,GammaPrior(1,1));
  txparams(4) = Parameter(1,GammaPrior(1,1));
  txparams(5) = Parameter(1,GammaPrior(1,1));
  txparams(6) = Parameter(1,GammaPrior(1,1));
  txparams(7) = Parameter(1,GammaPrior(1,1));
  txparams(8) = Parameter(1,GammaPrior(1,1));
  txparams(9) = Parameter(1,GammaPrior(1,1));
  txparams(10) = Parameter(1,GammaPrior(1,1));
  txparams(11) = Parameter(1,GammaPrior(1,1));

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1));

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,1);

//  BlockUpdate& infecUpdate = myMcmc->createBlockUpdate();
//  infecUpdate.push_back(txparams(0));
//  infecUpdate.push_back(txparams(1));
//  infecUpdate.push_back(txparams(4));
//  infecUpdate.push_back(txparams(5));
//  infecUpdate.push_back(txparams(6));
//  infecUpdate.push_back(txparams(7));
//
//  BlockUpdate& suscepUpdate = myMcmc->createBlockUpdate();
//  suscepUpdate.push_back(txparams(0));
//  suscepUpdate.push_back(txparams(1));
//  suscepUpdate.push_back(txparams(8));
//  suscepUpdate.push_back(txparams(9));
//  suscepUpdate.push_back(txparams(10));
//  suscepUpdate.push_back(txparams(11));
//
//  BlockUpdate& deltaUpdate = myMcmc->createBlockUpdate();
//  deltaUpdate.push_back(txparams(3));
//  deltaUpdate.push_back(txparams(2));

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/scratch/stsiab/fmdFullModelScalVar3.p" << comm.size() << ".parms";
  occFn << "/scratch/stsiab/fmdFullModelScalVar3.p" << comm.size() << ".occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[3]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  if(comm.rank() == 0) {
      cout << "Trans parm acceptance: " << acceptance["transParms"] << endl;
      //cout << "Alpha acceptance: " << acceptance["alpha"] << endl;
      //cout << "Alphastar acceptance: " << acceptance["alphastar"] << endl;
      //cout << "delta acceptance: " << acceptance["delta"] << endl;
      //cout << "epsilon acceptance: " << acceptance["epsilon"] << endl;
      cout << "Infection acceptance: " << acceptance["I"] << endl;
  }

  delete writer;
  delete myMcmc;
  delete myPopulation;

  MPI::Finalize();
  return EXIT_SUCCESS;

}
