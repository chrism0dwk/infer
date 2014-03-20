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
#include "GpuLikelihood.hpp"
#include "PosteriorHDF5Writer.hpp"

using namespace std;
using namespace boost::numeric;
using namespace EpiRisk;

#define NSPECIES 3
#define NEVENTS 3

bool doCompareProdVec = false;

void
sig_handler(int signo)
{
  if(signo == SIGUSR1) {
      std::cout << "Caught SIGUSR1" << std::endl;
      doCompareProdVec = true;
  }
}

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

class GaussianPrior : public Prior
{
  float mu_;
  float sigma_;
public:
  GaussianPrior(const float mu, const float sigma) :
    mu_(mu),sigma_(sigma)
  {
  };
  float
  operator()(const float x)
  {
    return gsl_ran_gaussian_pdf(x-mu_, sigma_);
  }
  Prior*
  create() const { return new GaussianPrior(mu_, sigma_); }
  Prior*
  clone() const { return new GaussianPrior(*this); }
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

  if (argc != 10)
    {
      cerr
          << "Usage: fmdMcmc <pop file> <epi file> <contact matrix> <output folder> <obs time> <num iterations> <seed> <nc percentage> <gpu>"
          << endl;
      return EXIT_FAILURE;
    }

  if(signal(SIGUSR1,sig_handler) == SIG_ERR)
    throw runtime_error("Cannot register signal handler");

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);
  ContactDataImporter* contactDataImporter = new ContactDataImporter(argv[3]);

  float obsTime = atof(argv[5]);
  size_t seed = atoi(argv[7]);
  float ncratio = atof(argv[8]);
  int gpuId = atoi(argv[9]);
  cout << "GPU: " << gpuId << endl;
  GpuLikelihood likelihood(*popDataImporter, *epiDataImporter, *contactDataImporter,
			   (size_t) 1, obsTime, 50.0f, false, gpuId);

  delete contactDataImporter;
  delete popDataImporter;
  delete epiDataImporter;

  // Parameters
  // Set up parameters
  Parameter epsilon1(2.7e-8, GammaPrior(2.7e-8, 1), "epsilon1");
  Parameter gamma1(1.0, GammaPrior(1, 1), "gamma1");
  Parameter delta(1.0, GammaPrior(1, 1), "delta");
  Parameter omega(1.2, GammaPrior(1,1), "omega");
  Parameter beta1(0.1, GammaPrior(1,1), "beta1");
  Parameter beta2(0.1, GammaPrior(1,1), "beta2");
  Parameter nu(9.0, GaussianPrior(-21.0, 15.3), "nu");
  Parameter alpha1(0.8f, BetaPrior(2, 2), "alpha1");
  Parameter alpha2(0.8f, BetaPrior(2, 2), "alpha2");
  Parameter a(4.0, GammaPrior(1, 1), "a");
  Parameter b(0.05, GammaPrior(2.5, 50), "b");
  Parameters phi(3);
  phi[0] = Parameter(1, GammaPrior(1,1), "phi0");
  phi[1] = Parameter(0.5, BetaPrior(20,20), "phi1");
  phi[2] = Parameter(0.2, BetaPrior(1,30), "phi2");

  likelihood.SetMovtBan(0.0f);
  likelihood.SetParameters(epsilon1, gamma1, phi, delta, omega, beta1, beta2,
			   nu, alpha1, alpha2, a, b);


  // Set up MCMC algorithm
  cout << "Initializing MCMC" << endl;
  Mcmc::Initialize();

  Mcmc::McmcRoot mcmc(likelihood, seed);

  UpdateBlock txDelta;
  txDelta.add(epsilon1);
  //txDelta.add(gamma1);
  txDelta.add(beta1);
  txDelta.add(beta2);
  txDelta.add(delta);
  txDelta.add(phi[1]);
  txDelta.add(phi[2]);
  //txDelta.add(nu);
  txDelta.add(alpha1);
  txDelta.add(alpha2);
  Mcmc::AdaptiveMultiLogMRW* updateDistance =
      (Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW",
          "txDistance");
  //Mcmc::AdaptiveSingleMRW* updateDistance =
    //  (Mcmc::AdaptiveSingleMRW*) mcmc.Create("AdaptiveSingleMRW","delta");
  updateDistance->SetParameters(txDelta);

  UpdateBlock updateNuBlk;
  updateNuBlk.add(nu);
  // Mcmc::AdaptiveSingleMRW* updateNu = (Mcmc::AdaptiveSingleMRW*) mcmc.Create("AdaptiveSingleMRW", "updateNu");
  // updateNu->SetParameters(updateNuBlk);

  UpdateBlock infecPeriod;
  infecPeriod.add(a);
  infecPeriod.add(b);
  Mcmc::InfectionTimeUpdate* updateInfecTime =
      (Mcmc::InfectionTimeUpdate*) mcmc.Create("InfectionTimeUpdate",
          "infecTimes");
  //updateInfecTime->SetCompareProductVector(&doCompareProdVec);
  updateInfecTime->SetParameters(infecPeriod);
  updateInfecTime->SetUpdateTuning(4.0);
  updateInfecTime->SetReps(800);
  updateInfecTime->SetOccults(true);

   UpdateBlock bUpdate; bUpdate.add(b);
   Mcmc::InfectionTimeGammaCentred* updateBC =
      (Mcmc::InfectionTimeGammaCentred*) mcmc.Create("InfectionTimeGammaCentred", "b_centred");
   updateBC->SetParameters(bUpdate);
   updateBC->SetTuning(0.014);

   Mcmc::InfectionTimeGammaNC* updateBNC =
      (Mcmc::InfectionTimeGammaNC*)mcmc.Create("InfectionTimeGammaNC", "b_ncentred");
   updateBNC->SetParameters(bUpdate);
   updateBNC->SetTuning(0.0007);
   updateBNC->SetNCRatio(1.0);

    //// Output ////

    // Make output directory
    string outputFile(argv[4]);
    PosteriorHDF5Writer output(outputFile, likelihood);
    output.AddParameter(epsilon1);
    output.AddParameter(gamma1);
    output.AddParameter(phi[1]);
    output.AddParameter(phi[2]);
    output.AddParameter(delta);
    output.AddParameter(omega);
    output.AddParameter(beta1);
    output.AddParameter(beta2);
    output.AddParameter(nu);
    output.AddParameter(alpha1);
    output.AddParameter(alpha2);
    output.AddParameter(b);

    boost::function< float () > getlikelihood = boost::bind(&GpuLikelihood::GetLogLikelihood, &likelihood);
    output.AddSpecial("loglikelihood",getlikelihood);
    boost::function< float () > getnuminfecs = boost::bind(&GpuLikelihood::GetNumInfecs, &likelihood);
    output.AddSpecial("numInfecs",getnuminfecs);
    boost::function< float () > getmeanI2N = boost::bind(&GpuLikelihood::GetMeanI2N, &likelihood);
    output.AddSpecial("meanI2N", getmeanI2N);
    boost::function< float () > getmeanOccI = boost::bind(&GpuLikelihood::GetMeanOccI, &likelihood);
    output.AddSpecial("meanOccI", getmeanOccI);


    // Run the chain
    cout << "Running MCMC" << endl;
    for(size_t k=0; k<atoi(argv[6]); ++k)
      {
        if(k % 100 == 0)
          {
            cout << "Iteration " << k << endl;
            output.flush();
          }
        mcmc.Update();
        output.write();
	likelihood.PrintLikelihoodComponents();
      }

    // Wrap up
    map<string, float> acceptance = mcmc.GetAcceptance();

    for(map<string, float>::const_iterator it = acceptance.begin();
        it != acceptance.end(); ++it)
      {
        cout << it->first << ": " << it->second << "\n";
      }

    cout << "Covariances\n";
    cout << updateDistance->GetCovariance() << "\n";

  return EXIT_SUCCESS;

}

