/*************************************************************************
 *  ./src/unitTests/fmdMcmc.cpp
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

  Parameters params(12);
  params(0) = Parameter(0.005,GammaPrior(1,1));
  params(1) = Parameter(0.004,GammaPrior(1,1));
  params(2) = Parameter(1.2,GammaPrior(1,1));
  params(3) = Parameter(2e-6,GammaPrior(1,1));
  params(4) = Parameter(1,GammaPrior(1,1));
  params(5) = Parameter(1,GammaPrior(1,1));
  params(6) = Parameter(1,GammaPrior(1,1));
  params(7) = Parameter(1,GammaPrior(1,1));
  params(8) = Parameter(1,GammaPrior(1,1));
  params(9) = Parameter(1,GammaPrior(1,1));
  params(10) = Parameter(1,GammaPrior(1,1));
  params(11) = Parameter(1,GammaPrior(1,1));

  Mcmc* myMcmc = new Mcmc(*myPopulation, params,1);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/scratch/stsiab/fmdFullModel.p" << comm.size() << ".parms";
  occFn << "/scratch/stsiab/fmdFullModel.p" << comm.size() << ".occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[3]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  if(comm.rank() == 0) {
      cout << "Alpha acceptance: " << acceptance["alpha"] << endl;
      cout << "Alphastar acceptance: " << acceptance["alphastar"] << endl;
      cout << "delta acceptance: " << acceptance["delta"] << endl;
      cout << "epsilon acceptance: " << acceptance["epsilon"] << endl;
      cout << "Infection acceptance: " << acceptance["I"] << endl;
  }

  delete writer;
  delete myMcmc;
  delete myPopulation;

  MPI::Finalize();
  return EXIT_SUCCESS;

}
