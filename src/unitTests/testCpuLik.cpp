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
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>

#include <gsl/gsl_randist.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "config.h"

#include "Data.hpp"
#include "Parameter.hpp"
#include "CpuLikelihood.hpp"

using namespace std;
using namespace boost::numeric;
using namespace EpiRisk;

#define NSPECIES 3
#define NEVENTS 3

inline
double
timeinseconds(const timeval a, const timeval b)
{
  timeval result;
  timersub(&b, &a, &result);
  return result.tv_sec + result.tv_usec / 1000000.0;
}

class GammaPrior : public Prior
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

class BetaPrior : public Prior
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


struct ParamSetting
{
  double value;
  double priorparams[2];
};

struct ParamSettings
{
  ParamSetting epsilon;
  ParamSetting gamma1;
  ParamSetting gamma2;
  ParamSetting delta;
  ParamSetting xi_p, xi_s;
  ParamSetting psi_c, psi_p, psi_s;
  ParamSetting zeta_p, zeta_s;
  ParamSetting phi_c, phi_p, phi_s;
  ParamSetting a, b;
};


int
main(int argc, char* argv[])
{
  // Tests out class Mcmc

  cerr << PACKAGE_NAME << " " << PACKAGE_VERSION << " compiled " << __DATE__
      << " " << __TIME__ << endl;

  if (argc != 4)
    {
      cerr
          << "Usage: testCpuLik <pop file> <epi file> <obs time>"
          << endl;
      return EXIT_FAILURE;
    }

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);
  float obsTime = atof(argv[3]);

  timeval start, end;
  cout << "Timing constructor..." << flush;
  gettimeofday(&start, NULL);
  CpuLikelihood likelihood(*popDataImporter, *epiDataImporter,
      (size_t) 3, obsTime, 25.0f);
  gettimeofday(&end, NULL);
  cout << "Done.\nConstructor took " << timeinseconds(start,end) << " seconds" << endl;

  delete popDataImporter;
  delete epiDataImporter;

  // Parameters
  // Set up parameters
  Parameter epsilon1(1.667e-6, GammaPrior(5e-5, 1), "epsilon1");
  Parameter epsilon2(1.0, GammaPrior(1,1), "epsilon2");
  Parameter gamma1(4.904e-05, GammaPrior(1, 1), "gamma1");
  Parameter gamma2(1.0, GammaPrior(2, 4), "gamma2");
  Parameters xi(3);
  xi[0] = Parameter(1.0, GammaPrior(1, 1), "xi_c");
  xi[1] = Parameter(0.417851, GammaPrior(1, 1), "xi_p");
  xi[2] = Parameter(11.395, GammaPrior(1, 1), "xi_s");
  Parameters psi(3);
  psi[0] = Parameter(0.86, BetaPrior(15, 15), "psi_c");
  psi[1] = Parameter(0.44, BetaPrior(15, 15), "psi_p");
  psi[2] = Parameter(0.39, BetaPrior(15, 15), "psi_s");
  Parameters zeta(3);
  zeta[0] = Parameter(1.0, GammaPrior(1, 1), "zeta_c");
  zeta[1] = Parameter(1.070e-03, GammaPrior(1, 1), "zeta_p");
  zeta[2] = Parameter(0.3368, GammaPrior(1, 1), "zeta_s");
  Parameters phi(3);
  phi[0] = Parameter(0.4061, BetaPrior(15, 15), "phi_c");
  phi[1] = Parameter(0.4365, BetaPrior(15, 15), "phi_p");
  phi[2] = Parameter(0.3372, BetaPrior(15, 15), "phi_s");
  Parameter delta(2.0, GammaPrior(1, 1), "delta");
  Parameter nu(0.001, GammaPrior(1, 1), "nu");
  Parameter alpha(4.0, GammaPrior(1, 1), "alpha");
  Parameter a(4.0, GammaPrior(1, 1), "a");
  Parameter b(0.5, GammaPrior(4.0, 8), "b");
  Parameter omega(1.5, GammaPrior(1,1), "omega");

  likelihood.SetMovtBan(22.0f);
  likelihood.SetParameters(epsilon1, epsilon2, gamma1, gamma2, xi, psi, zeta, phi, delta, omega,
      nu, alpha, a, b);
  
  cout << "Timing likelihood..." << endl;

  const size_t numreps = 1000;
  double timing[numreps];
  
  for(int i=0; i<numreps; ++i) {
    gettimeofday(&start, NULL);
    likelihood.FullCalculate();
    gettimeofday(&end, NULL);
    timing[i] = timeinseconds(start,end);
  }
  for(int i=1; i<numreps; ++i) timing[0] += timing[i];

  cout << "Likelihood took: " << timing[0]/(double)numreps << " seconds" << endl;
  cout << "Value: " << likelihood.GetLogLikelihood() << endl;
  likelihood.PrintLikelihoodComponents();

  likelihood.UpdateInfectionTime()




  return EXIT_SUCCESS;

}

