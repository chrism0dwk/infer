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
#include "Parameter.hpp"

using namespace std;
using namespace boost::numeric;

#define NSPECIES 3
#define NEVENTS 3

class GammaPrior : public EpiRisk::Prior
{
  float shape_;
  float rate_;
public:
  GammaPrior(const float shape, const float rate)
  {
    shape_ = shape;
    rate_ = rate;
  }
  float
  operator()(const float x)
  {
    return gsl_ran_gamma_pdf(x, shape_, 1 / rate_);
  }
  Prior*
  create() const
  {
    return new GammaPrior(shape_, rate_);
  }
  Prior*
  clone() const
  {
    return new GammaPrior(*this);
  }
};

class BetaPrior : public EpiRisk::Prior
{
  float a_;
  float b_;
public:
  BetaPrior(const float a, const float b) :
    a_(a), b_(b)
  {
  }
  ;
  float
  operator()(const float x)
  {
    return gsl_ran_beta_pdf(x, a_, b_);
  }
  Prior*
  create() const
  {
    return new BetaPrior(a_, b_);
  }
  Prior*
  clone() const
  {
    return new BetaPrior(*this);
  }
};



int main(int argc, char* argv[])
{
  // Tests out GpuLikelihood class

  if (argc != 5) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <dist matrix> <obsTime>" << endl;
      return EXIT_FAILURE;
  }

  float obsTime = atof(argv[4]);

  // Import  data
  PopDataImporter population(argv[1]);
  EpiDataImporter epidemic(argv[2]);
  DistMatrixImporter distance(argv[3]);

  // Set up GpuLikelihood
  GpuLikelihood* likelihood = new GpuLikelihood(population, epidemic, distance, NSPECIES, obsTime);


  // Set up parameters
  Parameter epsilon(7.72081e-05, GammaPrior(1, 1), "epsilon");
  Parameter gamma1(0.01, GammaPrior(1, 1), "gamma1");
  Parameter gamma2(0.5, GammaPrior(1, 1), "gamma2");
  Parameters xi(3);
  xi[0] = Parameter(1.0, GammaPrior(1, 1), "xi_c");
  xi[1] = Parameter(0.00205606, GammaPrior(1, 1), "xi_p");
  xi[2] = Parameter(0.613016, GammaPrior(1, 1), "xi_s");
  Parameters psi(3);
  psi[0] = Parameter(0.237344, BetaPrior(2, 2), "psi_c");
  psi[1] = Parameter(0.665464, BetaPrior(2, 2), "psi_p");
  psi[2] = Parameter(0.129998, BetaPrior(2, 2), "psi_s");
  Parameters zeta(3);
  zeta[0] = Parameter(1.0, GammaPrior(1, 1), "zeta_c");
  zeta[1] = Parameter(0.000295018, GammaPrior(1, 1), "zeta_p");
  zeta[2] = Parameter(0.259683, GammaPrior(1, 1), "zeta_s");
  Parameters phi(3);
  phi[0] = Parameter(0.402155, BetaPrior(2, 2), "phi_c");
  phi[1] = Parameter(0.749019, BetaPrior(2, 2), "phi_p");
  phi[2] = Parameter(0.365774, BetaPrior(2, 2), "phi_s");
  Parameter delta(1.14985, GammaPrior(1, 1), "delta");
  Parameter a(4.0, GammaPrior(1,1), "a");
  Parameter b(0.25, GammaPrior(1,1), "b");

  likelihood->SetParameters(epsilon,gamma1,gamma2,xi,psi,zeta,phi,delta, a, b);

  // Calculate
  likelihood->FullCalculate();



  // Fiddle with the population

  list<int> possibleOccults;
  for(size_t i=likelihood->GetNumInfecs(); i<likelihood->GetMaxInfecs(); ++i)
    possibleOccults.push_back(i);
  list<int> occults;

  gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
  gsl_rng_set(r, 3);


  likelihood->FullCalculate();

  for(size_t i=0; i<1000; ++i)
    {

      int toMove; int pos;
      float inTime;
      list<int>::iterator it;


      int chooseMove = gsl_rng_uniform_int(r,3);
      switch(chooseMove) {
      case 0:
        toMove = gsl_rng_uniform_int(r, likelihood->GetNumInfecs());
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
            likelihood->DeleteInfectionTime(likelihood->GetNumInfecs() + pos);
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

  delete likelihood;

  return EXIT_SUCCESS;

}
