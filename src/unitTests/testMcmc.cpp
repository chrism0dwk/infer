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

  Parameters* myParameters = new Parameters(0.4,0.5,0.6);

  Mcmc* myMcmc = new Mcmc(*myPopulation, *myParameters);

  char keyPress = 'c';
  while(keyPress != 'x') {
      //myMcmc->calcLogLikelihood();
      cout << "Likelihood: " << myMcmc->getLogLikelihood() << endl;
      cout << "\nPress a key (x to exit)...";
      cin >> keyPress;
  }

  delete myMcmc;
  delete myParameters;
  delete myPopulation;

  return EXIT_SUCCESS;

}
