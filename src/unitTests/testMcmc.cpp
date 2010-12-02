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
  myPopulation->setObsTime(34.405968);

  delete popDataImporter;
  delete epiDataImporter;

  Parameters* myParameters = new Parameters(4);
  (*myParameters)(0) = Parameter(0.03,GammaPrior(0.1,0.1));
  (*myParameters)(1) = Parameter(0.01,GammaPrior(0.1,0.1));
  (*myParameters)(2) = Parameter(0.2,GammaPrior(0.1,0.1));
  (*myParameters)(3) = Parameter(2e-6,GammaPrior(0.002,10000));

  Mcmc* myMcmc = new Mcmc(*myPopulation, *myParameters,1);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/scratch/stsiab/myParams1." << comm.size() << ".parms";
  occFn << "/scratch/stsiab/myOccults1." << comm.size() << ".occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[3]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  if(comm.rank() == 0) {
      cout << "Parameter acceptance: " << acceptance["transParms"] << endl;
      cout << "Infection acceptance: " << acceptance["I"] << endl;
  }

  delete writer;
  delete myMcmc;
  delete myParameters;
  delete myPopulation;

  MPI::Finalize();
  return EXIT_SUCCESS;

}
