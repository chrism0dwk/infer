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

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/bind.hpp>
namespace po = boost::program_options;

#include "config.h"
#include "Mcmc.hpp"
#include "McmcFactory.hpp"
#include "MCMCUpdater.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"
#include "Parameter.hpp"
#include "CpuVecLikelihood.hpp"
#include "PosteriorHDF5Writer.hpp"

using namespace std;
using namespace boost::numeric;
using namespace EpiRisk;

#define NSPECIES 3
#define NEVENTS 3

bool doCompareProdVec = false;

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

void
sig_handler(int signo)
{
  if(signo == SIGUSR1) {
      std::cout << "Caught SIGUSR1" << std::endl;
      doCompareProdVec = true;
  }
}

// inline
// double
// timeinseconds(const timeval a, const timeval b)
// {
//   timeval result;
//   timersub(&b, &a, &result);
//   return result.tv_sec + result.tv_usec / 1000000.0;
// }

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

class InfSuscSN : public StochasticNode
{
  Parameter* A_;
public:
  InfSuscSN(Parameter& A, Parameter& B) :
      A_(&A), StochasticNode(B)
  {
  }
  InfSuscSN*
  clone()
  {
    return new InfSuscSN(*this);
  }
  double
  getValue() const
  {
    return *A_ * *param_;
  }
  void
  setValue(const double x)
  {
    *param_ = x / *A_;
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

struct Settings
{
  string populationfile;
  string epidemicfile;
  string connectionfile;
  string posteriorfile;

  double obstime;
  size_t iterations;
  size_t iupdates;
  int seed;

  ParamSettings parameters;

  void
  load(const string& filename)
  {
    using boost::property_tree::ptree;

    typedef boost::property_tree::ptree_bad_data bad_data;
    typedef boost::property_tree::ptree_bad_path bad_xml;
    typedef boost::property_tree::ptree_error runtime_error;
    ptree pt;

    read_xml(filename, pt);

    populationfile = pt.get<string>("fmdMcmc.paths.population");
    epidemicfile = pt.get<string>("fmdMcmc.paths.epidemic");
    connectionfile = pt.get<string>("fmdMcmc.paths.connections");
    posteriorfile = pt.get<string>("fmdMcmc.paths.posterior");

    obstime = pt.get<double>("fmdMcmc.options.obstime", POSINF);
    iterations = pt.get<double>("fmdMcmc.options.iterations", 1);
    iupdates = pt.get<double>("fmdMcmc.options.iupdates", 0);
    seed = pt.get<int>("fmdMcmc.options.seed", 1);

    parameters.epsilon.value = pt.get<double>(
        "fmdMcmc.parameters.epsilon.value", 0.5);
    parameters.epsilon.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.epsilon.prior.gamma.a", 1);
    parameters.epsilon.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.epsilon.prior.gamma.b", 1);

    parameters.gamma1.value = pt.get<double>("fmdMcmc.parameters.gamma1.value",
        0.5);
    parameters.gamma1.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.gamma1.prior.gamma.a", 1);
    parameters.gamma1.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.gamma1.prior.gamma.b", 1);

    parameters.gamma2.value = pt.get<double>("fmdMcmc.parameters.gamma2.value",
        0.5);
    parameters.gamma2.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.gamma2.prior.gamma.a", 1);
    parameters.gamma2.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.gamma2.prior.gamma.b", 1);

    parameters.delta.value = pt.get<double>("fmdMcmc.parameters.delta.value",
        0.5);
    parameters.delta.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.delta.prior.gamma.a", 1);
    parameters.delta.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.delta.prior.gamma.b", 1);

    parameters.xi_p.value = pt.get<double>("fmdMcmc.parameters.xi_p.value",
        0.5);
    parameters.xi_p.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.xi_p.prior.gamma.a", 1);
    parameters.xi_p.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.xi_p.prior.gamma.b", 1);

    parameters.xi_s.value = pt.get<double>("fmdMcmc.parameters.xi_s.value",
        0.5);
    parameters.xi_s.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.xi_s.prior.gamma.a", 1);
    parameters.xi_s.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.xi_s.prior.gamma.b", 1);

    parameters.phi_c.value = pt.get<double>("fmdMcmc.parameters.phi_c.value",
        0.5);
    parameters.phi_c.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.phi_c.prior.beta.a", 1);
    parameters.phi_c.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.phi_c.prior.beta.b", 1);

    parameters.phi_p.value = pt.get<double>("fmdMcmc.parameters.phi_p.value",
        0.5);
    parameters.phi_p.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.phi_p.prior.beta.a", 1);
    parameters.phi_p.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.phi_p.prior.beta.b", 1);

    parameters.phi_s.value = pt.get<double>("fmdMcmc.parameters.phi_s.value",
        0.5);
    parameters.phi_s.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.phi_s.prior.beta.a", 1);
    parameters.phi_s.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.phi_s.prior.beta.b", 1);

    parameters.zeta_p.value = pt.get<double>("fmdMcmc.parameters.zeta_p.value",
        0.5);
    parameters.zeta_p.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.zeta_p.prior.gamma.a", 1);
    parameters.zeta_p.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.zeta_p.prior.gamma.b", 1);

    parameters.zeta_s.value = pt.get<double>("fmdMcmc.parameters.zeta_s.value",
        0.5);
    parameters.zeta_s.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.zeta_s.prior.gamma.a", 1);
    parameters.zeta_s.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.zeta_s.prior.gamma.b", 1);

    parameters.psi_c.value = pt.get<double>("fmdMcmc.parameters.psi_c.value",
        0.5);
    parameters.psi_c.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.psi_c.prior.beta.a", 1);
    parameters.psi_c.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.psi_c.prior.beta.b", 1);

    parameters.psi_p.value = pt.get<double>("fmdMcmc.parameters.psi_p.value",
        0.5);
    parameters.psi_p.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.psi_p.prior.beta.a", 1);
    parameters.psi_p.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.psi_p.prior.beta.b", 1);

    parameters.psi_s.value = pt.get<double>("fmdMcmc.parameters.psi_s.value",
        0.5);
    parameters.psi_s.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.psi_s.prior.beta.a", 1);
    parameters.psi_s.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.psi_s.prior.beta.b", 1);

    parameters.a.value = pt.get<double>("fmdMcmc.parameter.a.value", 0.08);
    parameters.b.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.a.prior.gamma.a", 1);
    parameters.b.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.a.prior.gamma.b", 1);

    parameters.b.value = pt.get<double>("fmdMcmc.parameter.b.value", 0.005);
    parameters.b.priorparams[0] = pt.get<double>(
        "fmdMcmc.parameters.b.prior.gamma.a", 1);
    parameters.b.priorparams[1] = pt.get<double>(
        "fmdMcmc.parameters.b.prior.gamma.b", 1);

  }
};

int
main(int argc, char* argv[])
{
  // Tests out class Mcmc

  cerr << PACKAGE_NAME << " " << PACKAGE_VERSION << " compiled " << __DATE__
      << " " << __TIME__ << endl;

  if (argc != 9)
    {
      cerr
          << "Usage: fmdMcmc <pop file> <epi file> <output folder> <obs time> <num iterations> <seed> <nc percentage> <gpu>"
          << endl;
      return EXIT_FAILURE;
    }

  if(signal(SIGUSR1,sig_handler) == SIG_ERR)
    throw runtime_error("Cannot register signal handler");

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);
  float obsTime = atof(argv[4]);
  size_t seed = atoi(argv[6]);
  int gpuId = atoi(argv[8]);

  CpuLikelihood likelihood(*popDataImporter, *epiDataImporter,
			   (size_t) 3, obsTime, 25.0f, false);

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
  Parameter delta(0.2431, GammaPrior(1, 1), "delta");
  Parameter nu(0.001, GammaPrior(1, 1), "nu");
  Parameter alpha(60, GammaPrior(1, 1), "alpha");
  Parameter a(4.0, GammaPrior(1, 1), "a");
  Parameter b(0.5, GammaPrior(4.0, 8), "b");
  Parameter omega(1.5, GammaPrior(1,1), "omega");

  likelihood.SetMovtBan(22.0f);
  likelihood.SetParameters(epsilon1, epsilon2, gamma1, gamma2, xi, psi, zeta, phi, delta, omega,
      nu, alpha, a, b);

  // Set up MCMC algorithm
  cout << "Initializing MCMC" << endl;
  Mcmc::Initialize();

  Mcmc::McmcRoot mcmc(likelihood, seed);

  float ncratio = atof(argv[7]);

  UpdateBlock txDelta;
  txDelta.add(epsilon1);
  txDelta.add(epsilon2);
  txDelta.add(gamma1);
  txDelta.add(gamma2);
  txDelta.add(delta);
  //txDelta.add(nu);
  //txDelta.add(alpha);
  Mcmc::AdaptiveMultiLogMRW* updateDistance =
      (Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW",
          "txDistance");
  updateDistance->SetParameters(txDelta);

  UpdateBlock txPsi;
  txPsi.add(psi[0]);
  txPsi.add(psi[1]);
  txPsi.add(psi[2]);
  Mcmc::AdaptiveMultiLogMRW* updatePsi =
      (Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW", "txPsi");
  updatePsi->SetParameters(txPsi);

  UpdateBlock txPhi;
  txPhi.add(phi[0]);
  txPhi.add(phi[1]);
  txPhi.add(phi[2]);
  Mcmc::AdaptiveMultiLogMRW* updatePhi =
      (Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW", "txPhi");
  updatePhi->SetParameters(txPhi);

  UpdateBlock txInfec;
  txInfec.add(gamma1);
  txInfec.add(xi[1]);
  txInfec.add(xi[2]);

  Mcmc::InfectivityMRW* updateInfec = (Mcmc::InfectivityMRW*) mcmc.Create(
     "InfectivityMRW", "txInfec");
  updateInfec->SetParameters(txInfec);

  UpdateBlock txSuscep;
  txSuscep.add(gamma1);
  txSuscep.add(zeta[1]);
  txSuscep.add(zeta[2]);
  Mcmc::SusceptibilityMRW* updateSuscep =
     (Mcmc::SusceptibilityMRW*) mcmc.Create("SusceptibilityMRW", "txSuscep");
  updateSuscep->SetParameters(txSuscep);

  // AdaptiveMultiMRW* updateDistanceLin = mcmc.NewAdaptiveMultiMRW("txDistanceLin",txDelta, 300);

  UpdateBlock infecPeriod;
  infecPeriod.add(a);
  infecPeriod.add(b);
  Mcmc::InfectionTimeUpdate* updateInfecTime =
      (Mcmc::InfectionTimeUpdate*) mcmc.Create("InfectionTimeUpdate",
          "infecTimes");
  //updateInfecTime->SetCompareProductVector(&doCompareProdVec);
  updateInfecTime->SetParameters(infecPeriod);
  updateInfecTime->SetUpdateTuning(2.5);
  updateInfecTime->SetReps(900);
  updateInfecTime->SetOccults(false);

  UpdateBlock bUpdate; bUpdate.add(b);
  Mcmc::InfectionTimeGammaCentred* updateBC =
      (Mcmc::InfectionTimeGammaCentred*) mcmc.Create("InfectionTimeGammaCentred", "b_centred");
  updateBC->SetParameters(bUpdate);
  updateBC->SetTuning(0.014);

  Mcmc::InfectionTimeGammaNC* updateBNC =
      (Mcmc::InfectionTimeGammaNC*)mcmc.Create("InfectionTimeGammaNC", "b_ncentred");
  updateBNC->SetParameters(bUpdate);
  updateBNC->SetTuning(0.0007);
  updateBNC->SetNCRatio(ncratio);

    //// Output ////

    // Make output directory
    string outputFile(argv[3]);
    PosteriorHDF5Writer output(outputFile, likelihood);
    output.AddParameter(epsilon1); output.AddParameter(epsilon2);
    output.AddParameter(gamma1);
    output.AddParameter(gamma2);  output.AddParameter(xi[0]);
    output.AddParameter(xi[1]);   output.AddParameter(xi[2]);
    output.AddParameter(psi[0]);  output.AddParameter(psi[1]);
    output.AddParameter(psi[2]);  output.AddParameter(zeta[0]);
    output.AddParameter(zeta[1]); output.AddParameter(zeta[2]);
    output.AddParameter(phi[0]);  output.AddParameter(phi[1]);
    output.AddParameter(phi[2]);  output.AddParameter(delta);
    output.AddParameter(nu);      output.AddParameter(alpha);
    output.AddParameter(b);

    boost::function< float () > getlikelihood = boost::bind(&Likelihood::GetLogLikelihood, &likelihood);
    output.AddSpecial("loglikelihood",getlikelihood);
    boost::function< float () > getnuminfecs = boost::bind(&Likelihood::GetNumInfecs, &likelihood);
    output.AddSpecial("numInfecs",getnuminfecs);
    boost::function< float () > getmeanI2N = boost::bind(&Likelihood::GetMeanI2N, &likelihood);
    output.AddSpecial("meanI2N", getmeanI2N);
    boost::function< float () > getmeanOccI = boost::bind(&Likelihood::GetMeanOccI, &likelihood);
    output.AddSpecial("meanOccI", getmeanOccI);

    // Output the population id index
    string idxfn = outputFile + ".stridx";
    ofstream idxfile; idxfile.open(idxfn.c_str(), ios::out);
    std::vector<std::string> ids; likelihood.GetIds(ids);
    for(std::vector<std::string>::const_iterator it = ids.begin();
    		it != ids.end();
    		it++)
    {
    	idxfile << *it << "\n";
    }
    idxfile.close();

    // Run the chain

    timespec start, end;
    cout << "Running MCMC" << endl;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID,&start);
    for(size_t k=0; k<atoi(argv[5]); ++k)
      {
        // if(k % 100 == 0)
        //   {
        //     cout << "Iteration " << k << endl;
        //     output.flush();
        //   }

        mcmc.Update();
        output.write();
      }
    clock_gettime(CLOCK_THREAD_CPUTIME_ID,&end);

    // Wrap up
    map<string, float> acceptance = mcmc.GetAcceptance();

    for(map<string, float>::const_iterator it = acceptance.begin();
        it != acceptance.end(); ++it)
      {
        cout << it->first << ": " << it->second << "\n";
      }

    cout << "\n\nTime taken: " << timediff(start,end) << endl;

  return EXIT_SUCCESS;

}

