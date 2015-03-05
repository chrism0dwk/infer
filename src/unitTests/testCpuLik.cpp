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
#include <time.h>
#include <signal.h>
#include <unistd.h>

#include <gsl/gsl_randist.h>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "config.h"
#include "Random.hpp"
#include "Data.hpp"
#include "Parameter.hpp"
#include "CpuLikelihood.hpp"
#include "Mcmc.hpp"
#include "McmcFactory.hpp"
#include "MCMCUpdater.hpp"

using namespace std;
using namespace boost::numeric;
using namespace EpiRisk;

#define NSPECIES 3
#define NEVENTS 3

#define THECLOCK CLOCK_THREAD_CPUTIME_ID

fp_t timediff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return (temp.tv_sec*1000000000L + temp.tv_nsec)/1000000000.0;
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

  timespec start;
  timespec end;
  cout << "Timing constructor..." << flush;
  clock_gettime(THECLOCK, &start);
  CpuLikelihood cpu(*popDataImporter, *epiDataImporter,
      (size_t) 3, obsTime, 25.0f);
  clock_gettime(THECLOCK, &end);
  cout << "Done.\nConstructor took " << timediff(start, end) << " seconds" << endl;

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
  Parameter alpha(4, GammaPrior(1, 1), "alpha");
  Parameter a(4.0, GammaPrior(1, 1), "a");
  Parameter b(0.5, GammaPrior(4.0, 8), "b");
  Parameter omega(1.5, GammaPrior(1,1), "omega");

  cpu.SetMovtBan(22.0f);
  cpu.SetParameters(epsilon1, epsilon2, gamma1, gamma2, xi, psi, zeta, phi, delta, omega,
      nu, alpha, a, b);
  
  cout << "Timing likelihood..." << flush;
  clock_gettime(THECLOCK, &start);
  cpu.FullCalculate();
  clock_gettime(THECLOCK, &end);

  cout << "Likelihood took: " << timediff(start, end) << " nanoseconds" << endl;
  cout << "Value: " << cpu.GetLogLikelihood() << endl;
  cpu.PrintLikelihoodComponents();


  // Timing
  const size_t reps = 1000;
  double avgtime = 0.0;
  Random rng(0);

  // Full calculation first
  for(size_t i=0; i<reps; ++i)
    {
      clock_gettime(THECLOCK, &start);
      cpu.FullCalculate();
      clock_gettime(THECLOCK, &end);
      avgtime += (timediff(start, end) - avgtime)/(i+1);
      cout << "." << flush;
    }
  cout << "\nCPU Vec FullCalculate: " << avgtime << " nanoseconds" << endl;

  // Moving
  avgtime = 0.0;
  for(size_t i=0; i<reps; ++i)
    {

      size_t idx = rng.integer(cpu.GetNumInfecs());
      fp_t d = rng.gamma(4, 0.5);
      clock_gettime(THECLOCK, &start);
      cpu.UpdateInfectionTime(idx, d);
      clock_gettime(THECLOCK, &end);
      avgtime += (timediff(start,end) - avgtime)/(i+1);
    }
  cout << "CPU Vec Update: " << avgtime << " nanoseconds" << endl;
  cpu.FullCalculate();
  avgtime = 0.0;
  for(size_t i=0; i<reps; ++i)
    {
      size_t idx = rng.integer(cpu.GetNumPossibleOccults());
      fp_t d = rng.gamma(4, 0.5);
      clock_gettime(THECLOCK, &start);
      cpu.AddInfectionTime(idx, d);
      clock_gettime(THECLOCK, &end);
      avgtime += (timediff(start, end) - avgtime)/(i+1);
    }
  cout << "CPU Vec Add: " << avgtime << " nanoseconds" << endl;
  cpu.FullCalculate();
  avgtime = 0.0;
  for(size_t i=0; i<reps; ++i)
    {
      size_t idx = rng.integer(cpu.GetNumOccults());
      clock_gettime(THECLOCK, &start);
      cpu.DeleteInfectionTime(idx);
      clock_gettime(THECLOCK, &end);
      avgtime += (timediff(start, end) - avgtime)/(i+1);
    }
  cout << "CPU Vec Del: " << avgtime << " nanoseconds" << endl;

  cout << "Likelihood:\n" << endl;
  cout << "\n\nMove: " << cpu.GetLogLikelihood() << endl;
  cpu.PrintLikelihoodComponents();
  cpu.FullCalculate();
  cout << "\n\nFull calc: " << cpu.GetLogLikelihood() << endl;
  cpu.PrintLikelihoodComponents();


  return EXIT_SUCCESS;

}

