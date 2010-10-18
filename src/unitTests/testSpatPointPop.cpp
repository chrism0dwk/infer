/***************************************************************************
 *   Copyright (C) 2010 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#include <iostream>
#include "Individual.hpp"
#include "SpatPointPop.hpp"
#include "Data.hpp"

using namespace EpiRisk;




int main(int argc, char* argv[])
{
  // Unit test for SpatPointPop

  if (argc != 3) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file>" << endl;
      return EXIT_FAILURE;
  }

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  myPopulation->setObsTime(10.0);

  delete popDataImporter;
  delete epiDataImporter;

  Population<TestCovars>::const_iterator iter = myPopulation->begin();
  while(iter != myPopulation->end()) {
      cout << iter->getId() << "\t"
           << iter->getCovariates()->x << "\t"
           << iter->getCovariates()->y << "\t"
           << iter->getCovariates()->herdSize << "\t"
           << iter->getI() << "\t"
           << iter->getN() << "\t"
           << iter->getR() << endl;
      iter++;
  }

  cout << "Total pop size: " << myPopulation->size() << endl;
  cout << "Num infections: " << myPopulation->numInfected() << endl;
  cout << "Num susceptibles: " << myPopulation->numSusceptible() << endl;

  Population<TestCovars>::PopulationIndex::Iterator infec = myPopulation->infecBegin();

  cout << "I1 id: " << infec->getId() << " " << infec->getI() <<  endl;
  infec++;
  cout << "I2 id: " << infec->getId() << " " << infec->getI() << endl;
  myPopulation->moveInfectionTime(2,10.0);
  cout << "Moved I1 to 10.0" << endl;

  infec = myPopulation->infecBegin();
  cout << "I1 id: " << infec->getId() << " " << infec->getI() <<  endl;
  infec++;
  cout << "I2 id: " << infec->getId() << " " << infec->getI() << endl;

  iter = myPopulation->begin();
  while(iter != myPopulation->end()) {
      cout << iter->getId() << "\t"
           << iter->getCovariates()->x << "\t"
           << iter->getCovariates()->y << "\t"
           << iter->getCovariates()->herdSize << "\t"
           << iter->getI() << "\t"
           << iter->getN() << "\t"
           << iter->getR() << endl;
      iter++;
  }




  delete myPopulation;
}
