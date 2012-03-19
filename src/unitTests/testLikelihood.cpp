/*************************************************************************
 *  ./src/unitTests/testLikelihood.cpp
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
 * testLikelihood.cpp
 *
 *  Created on: 4 Sep 2011
 *      Author: stsiab
 */

/*
 * testMcmc.cpp
 *
 *  Created on: Oct 15, 2010
 *      Author: stsiab
 */

#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "Data.hpp"
#include "GpuLikelihood.hpp"

using namespace std;
using namespace boost::numeric;

#define NSPECIES 3
#define NEVENTS 3

enum DiseaseStatus
{
  IP = 0,
  DC = 1,
  SUSC = 2
};


struct Covars
{
  string id;
  DiseaseStatus status;
  float I;
  float N;
  float R;
  float cattle;
  float pigs;
  float sheep;
};

typedef vector<Covars> Population;


struct IsId
{
  IsId(string id) : id_(id) {};
  bool
  operator()(const Covars& cov) const
  {
    return cov.id == id_;
  }
private:
  string id_;
};

struct CompareByI
{
  bool
  operator()(const Covars& lhs, const Covars& rhs) const
  {
    return lhs.I < rhs.I;
  }
};

struct CompareByStatus
{
  bool
  operator()(const Covars& lhs, const Covars& rhs) const
  {
    return (int)lhs.status < (int)rhs.status;
  }
};


void
importPopData(Population& population, const char* filename, map<string,size_t>& idMap)
{
  PopDataImporter importer(filename);
  idMap.clear();

  importer.open();
  try
    {
      size_t idx = 0;
      while (1)
        {
          PopDataImporter::Record record = importer.next();
          Covars covars;
          covars.id = record.id;
          covars.status = SUSC;
          covars.I = EpiRisk::POSINF;
          covars.N = EpiRisk::POSINF;
          covars.R = EpiRisk::POSINF;
          covars.cattle = record.data.cattle;
          covars.pigs = record.data.pigs;
          covars.sheep = record.data.sheep;
          idMap.insert(make_pair(covars.id, idx)); idx++;
          population.push_back(covars);
        }
    }
  catch (EpiRisk::fileEOF& e)
    {
      return;
    }

  importer.close();

}

void
importEpiData(Population& population, const char* filename, const float obsTime, map<string, size_t>& idMap, size_t* numCulled)
{
  EpiDataImporter importer(filename);
  size_t _numCulled = 0;

  importer.open();
  try
    {
      while (1)
        {
          EpiDataImporter::Record record = importer.next();
          map<string, size_t>::const_iterator map = idMap.find(record.id);
          if (map == idMap.end())
            throw range_error(
                "Key in epidemic data not found in population data");

          Population::iterator ref = population.begin() + map->second;
          // Check type
          if(record.data.I == EpiRisk::POSINF) ref->status = DC;
          else ref->status = IP;

          // Check data integrity
          if (record.data.N > record.data.R) {
              cerr << "Individual " << record.id << " has N > R.  Setting N = R\n";
              record.data.N = record.data.R;
          }
          if (record.data.R < record.data.I and record.data.I != EpiRisk::POSINF)
            {
              cerr << "WARNING: Individual " << record.id << " has I > R!  Setting I = R-7\n";
              record.data.I = record.data.R - 7;
            }

          ref->I = record.data.I; ref->N = record.data.N; ref->R = record.data.R;

          ref->R = min(ref->R, obsTime);
          ref->N = min(ref->N, ref->R);
          ref->I = min(ref->I, ref->N);

          _numCulled++;
        }
    }
  catch (EpiRisk::fileEOF& e)
    {
      *numCulled = _numCulled;
      return;
    }

  importer.close();
}


int main(int argc, char* argv[])
{
  // Tests out GpuLikelihood class

  if (argc != 5) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <dist matrix> <obsTime>" << endl;
      return EXIT_FAILURE;
  }

  float obsTime = atof(argv[4]);

  Population population;
  size_t numCulled;
  size_t numInfecs;
  map<string, size_t> idMap;

  // Import population data
  importPopData(population, argv[1], idMap);
  importEpiData(population, argv[2], obsTime, idMap, &numCulled);

  // Sort individuals by disease status (IPs -> DCs -> SUSCs)
  sort(population.begin(), population.end(), CompareByStatus());
  Covars cmp; cmp.status = DC;
  Population::iterator topOfIPs = lower_bound(population.begin(), population.end(), cmp, CompareByStatus());
  numInfecs = topOfIPs - population.begin();
  sort(population.begin(), topOfIPs, CompareByI());

  cout << "Population size: " << population.size() << "\n";
  cout << "Num infecs: " << numInfecs << "\n";
  cout << "Num culled: " << numCulled << "\n";

  // Rebuild population ID index
  idMap.clear();
  Population::const_iterator it = population.begin();
  for(size_t i=0; i<population.size(); i++)
    {
      idMap.insert(make_pair(it->id,i));
      it++;
    }

  // Import distances
  ublas::mapped_matrix<float> Dimport(numCulled, population.size());
  DistMatrixImporter* distMatrixImporter = new DistMatrixImporter(argv[3]);
  try {
      distMatrixImporter->open();
      while(1)
        {
          DistMatrixImporter::Record record = distMatrixImporter->next();
          map<string,size_t>::const_iterator i = idMap.find(record.id);
          map<string,size_t>::const_iterator j = idMap.find(record.data.j);
          if(i == idMap.end() or j == idMap.end())
            throw range_error("Key pair not found in population");
          if (i != j)
            Dimport(i->second, j->second) = record.data.distance*record.data.distance;
        }
  }
  catch (EpiRisk::fileEOF& e)
  {
      cout << "Imported " << Dimport.nnz() << " distance elements" << endl;
  }
  catch (exception& e)
  {
      throw e;
  }

  delete distMatrixImporter;

  // Set up GpuLikelihood
  GpuLikelihood* likelihood = new GpuLikelihood(population.size(), population.size(), numInfecs, numCulled, NSPECIES, obsTime, Dimport.nnz());

  // Set up Species and events
  float* speciesMatrix = new float[population.size()*NSPECIES];
  float* eventsMatrix = new float[population.size()*NEVENTS];
  it = population.begin();
  for(size_t i=0; i<population.size(); ++i)
    {
      speciesMatrix[i] = it->cattle;
      speciesMatrix[i+population.size()] = it->pigs;
      speciesMatrix[i+population.size()*2] = it->sheep;

      eventsMatrix[i] = it->I;
      eventsMatrix[i+population.size()] = it->N;
      eventsMatrix[i+population.size()*2] = it->R;
      ++it;
    }

  likelihood->SetEvents(eventsMatrix);
  likelihood->SetSpecies(speciesMatrix);

  // Set up distance matrix
  ublas::compressed_matrix<float>* D = new ublas::compressed_matrix<float>(Dimport);
  int* rowPtr = new int[D->index1_data().size()];
  for(size_t i=0; i<D->index1_data().size(); ++i) rowPtr[i] = D->index1_data()[i];
  int* colInd = new int[D->index2_data().size()];
  for(size_t i=0; i<D->index2_data().size(); ++i) colInd[i] = D->index2_data()[i];
  likelihood->SetDistance(D->value_data().begin(), rowPtr, colInd);

  delete[] rowPtr;
  delete[] colInd;
  delete D;

  // Set up parameters

  float epsilon = 7.72081e-05;
  float gamma1 = 0.01;
  float gamma2 = 0.0;
  float xi[] = {1.0, 0.00205606, 0.613016};
  float psi[] = {0.237344, 0.665464, 0.129998};
  float zeta[] = {1.0, 0.000295018, 0.259683};
  float phi[] = {0.402155, 0.749019, 0.365774};
  float delta = 1.14985;

  likelihood->SetParameters(&epsilon,&gamma1,&gamma2,xi,psi,zeta,phi,&delta);

  // Calculate
  likelihood->FullCalculate();



  // Fiddle with the population

  list<int> possibleOccults;
  for(size_t i=numInfecs; i<numCulled; ++i)
    possibleOccults.push_back(i);
  list<int> occults;

  gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
  gsl_rng_set(r, 3);
//  list<int>::iterator iter;
//  for(size_t i=0; i<1000; ++i) {
//      iter = possibleOccults.begin();
//      advance(iter,gsl_rng_uniform_int(r, possibleOccults.size()));
//      float inTime = gsl_ran_gamma(r, 1, 0.1);
//      cout << "Adding " << *iter << endl;
//      likelihood->LazyAddInfecTime(*iter, inTime);
//      occults.push_back(*iter);
//      possibleOccults.erase(iter);
//  }

  likelihood->FullCalculate();

  for(size_t i=0; i<1000; ++i)
    {

      int toMove; int pos;
      float inTime;
      list<int>::iterator it;


      int chooseMove = gsl_rng_uniform_int(r,3);
      switch(chooseMove) {
      case 0:
        toMove = gsl_rng_uniform_int(r, numInfecs);
        inTime = gsl_ran_gamma(r, 10, 1);
        likelihood->UpdateInfectionTime(toMove,inTime);
        break;
      case 1:
        it = possibleOccults.begin();
        advance(it,gsl_rng_uniform_int(r, possibleOccults.size()));
        inTime = gsl_ran_gamma(r, 1, 0.1);
        cout << "Adding " << *it << endl;
        likelihood->AddInfectionTime(*it, inTime);
        occults.push_back(*it);
        possibleOccults.erase(it);
        break;
      case 2:
        if (occults.size() > 0) {
            pos = gsl_rng_uniform_int(r, occults.size());
            it = occults.begin();
            advance(it, pos);
            cout << "Deleting " << pos << endl;;
            likelihood->DeleteInfectionTime(numInfecs + pos);
            possibleOccults.push_back(*it);
            occults.erase(it);
        }
        break;


      }

      cerr << "Num occults = " << occults.size() << endl;
      //likelihood->Calculate();
    }
  gsl_rng_free(r);

  likelihood->Calculate();
//  likelihood->AddInfectionTime(413, 5); likelihood->Calculate();
//  likelihood->AddInfectionTime(1000,7.5); likelihood->Calculate();
//  likelihood->DeleteInfectionTime(numInfecs); likelihood->Calculate();
//  likelihood->DeleteInfectionTime(numInfecs); likelihood->Calculate();



  delete likelihood;

  return EXIT_SUCCESS;

}
