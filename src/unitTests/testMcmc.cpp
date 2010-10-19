/*
 * testMcmc.cpp
 *
 *  Created on: Oct 15, 2010
 *      Author: stsiab
 */

#include <iostream>

#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"


int main(int argc, char* argv[])
{
  // Tests out class Mcmc

  if (argc != 3) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file>" << endl;
      return EXIT_FAILURE;
  }

  typedef Population<TestCovars> MyPopulation;

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  myPopulation->setObsTime(100.0);

  delete popDataImporter;
  delete epiDataImporter;

  Parameters* myParameters = new Parameters();

  Mcmc* myMcmc = new Mcmc(*myPopulation, *myParameters);

  for (size_t i=0; i<30; ++i) {
    myMcmc->calcLogLikelihood();
    cout << i << endl;
  }

  cout << "Likelihood: " << myMcmc->getLogLikelihood() << endl;


  delete myMcmc;
  delete myParameters;
  delete myPopulation;

  return EXIT_SUCCESS;

}
