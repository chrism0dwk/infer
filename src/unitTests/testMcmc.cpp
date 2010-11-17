/*************************************************************************
 *  ./src/unitTests/testMcmc.cpp
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
#include <mpi.h>

#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"


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

  MPI::Init(argc,argv);
  MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);

  if (argc != 3) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file>" << endl;
      return EXIT_FAILURE;
  }

  typedef Population<TestCovars> MyPopulation;

  // TODO: Required to parallelise the writers!

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
  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>("myParams.parms","myOccults.occ");

  map<string,double> acceptance = myMcmc->run(10000, *writer);

  if(MPI::COMM_WORLD.Get_rank() == 0) {
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
