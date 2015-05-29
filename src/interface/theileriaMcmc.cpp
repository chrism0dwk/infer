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
#include <unistd.h>
#include <boost/bind.hpp>

#include "Mcmc.hpp"
#include "McmcFactory.hpp"
#include "MCMCUpdater.hpp"
#include "Data.hpp"
#include "Parameter.hpp"
#include "GpuLikelihood.hpp"
#include "PosteriorHDF5Writer.hpp"
#include "RData.hpp"

#include "theileriaMcmc.hpp"

#define NSPECIES 3
#define NEVENTS 3

#define RNUM(x) Rcpp::NumericVector(x)

using namespace EpiRisk;

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

class UniformPrior : public Prior
{
private:
  float a_;
  float b_;
public:
  UniformPrior(const float a, const float b) : a_(a),b_(b)
  {
  };
  float operator()(const float x)
  {
    if(a_ <= x and x <= b_) return 1.0f/(b_-a_);
    else return 0.0f;
  }
  Prior*
  create() const { return new UniformPrior(a_, b_); }
  Prior*
  clone() const { return new UniformPrior(*this); }
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





RcppExport SEXP
TheileriaMcmc(SEXP population,
	      SEXP epidemic,
	      SEXP contact,
	      const SEXP ticks,
	      const SEXP obsTime,
	      const SEXP init,
	      const SEXP priorParms,
	      const SEXP control,
	      const SEXP outputfile)
{

  Rcpp::DataFrame       _population(population);
  Rcpp::DataFrame       _epidemic(epidemic);
  Rcpp::DataFrame       _contact(contact);
  Rcpp::DataFrame       _ticks(ticks);
  Rcpp::NumericVector   _obsTime(obsTime);
  Rcpp::List            _init(init);
  Rcpp::List            _priorParms(priorParms);
  Rcpp::List            _control(control);
  Rcpp::CharacterVector _outputFile(outputfile);

  Rcpp::IntegerVector numIter = _control["n.iter"];
  Rcpp::IntegerVector gpuId   = _control["gpuid"];
  Rcpp::IntegerVector seed    = _control["seed"];
  Rcpp::NumericVector tuneI   = _control["tune.I"];
  Rcpp::IntegerVector repsI   = _control["reps.I"];
  Rcpp::NumericVector dLimit  = _control["dlimit"];

  // // Load surveillance data
  list<TickSurv> tickdata;
  int maxTLAId = 0;
  Rcpp::IntegerVector region = _ticks["region"];
  Rcpp::NumericVector numpos = _ticks["numpos"];
  Rcpp::NumericVector total  = _ticks["total"];
  Rcpp::NumericVector ticka  = _ticks["a"];
  Rcpp::NumericVector tickb  = _ticks["b"];

  for(size_t i=0; i<_ticks.size(); ++i) {
    TickSurv t;
    t.region = region[i];
    t.numpos = numpos[i];
    t.total = total[i];
    t.a = ticka[i];
    t.b = tickb[i];
    tickdata.push_back(t);
    if(maxTLAId < t.region) maxTLAId = t.region;
  }


  // Create R -> GPU data adaptors
  PopRImporter* popRImporter = new PopRImporter(_population);
  EpiRImporter* epiRImporter = new EpiRImporter(_epidemic);
  ContactRImporter* contactRImporter = new ContactRImporter(_contact);

  // Initialize likelihood
  int gpu = gpuId[0];
  GpuLikelihood likelihood(*popRImporter, *epiRImporter, *contactRImporter,
  			   (size_t) 1, (float)_obsTime[0], 50.0f, false, gpu);

  delete contactRImporter;
  delete popRImporter;
  delete epiRImporter;


  // Initialize parameters
  Rcpp::NumericVector prior = _priorParms["epsilon1"];
  Rcpp::NumericVector startval = _init["epsilon1"];
  Parameter epsilon(startval[0], GammaPrior(prior[0], prior[1]), "epsilon");
  prior = _priorParms["delta"]; startval = _init["delta"];
  Parameter delta(startval[0], GammaPrior(prior[0], prior[1]), "delta");
  prior = _priorParms["omega"]; startval = _init["omega"];
  Parameter omega(startval[0], GammaPrior(prior[0],prior[1]), "omega");
  prior = _priorParms["beta1"]; startval = _init["beta1"];
  Parameter beta1(startval[0], GammaPrior(prior[0],prior[1]), "beta1");
  prior = _priorParms["beta2"]; startval = _init["beta2"];
  Parameter beta2(startval[0], GammaPrior(prior[0],prior[1]), "beta2");
  startval = _init["nu"];
  Parameter nu(startval[0], GaussianPrior(1, 1), "nu");
  prior = _priorParms["alpha1"]; startval = _init["alpha1"];
  Parameter alpha1(startval[0], BetaPrior(prior[0], prior[1]), "alpha1");
  prior = _priorParms["alpha2"]; startval = _init["alpha2"];
  Parameter alpha2(startval[0], BetaPrior(prior[0], prior[1]), "alpha2");
  prior = _priorParms["alpha3"]; startval = _init["alpha3"];
  Parameter alpha3(startval[0], GammaPrior(prior[0], prior[1]), "alpha3");
  prior = _priorParms["zeta"]; startval = _init["zeta"];
  Parameter zeta(startval[0], GammaPrior(prior[0],prior[1]), "zeta");
  startval = _init["a"];
  Parameter a(startval[0], GammaPrior(1, 1), "a");
  prior = _priorParms["b"]; startval = _init["b"];
  Parameter b(startval[0], GammaPrior(prior[0],prior[1]), "b");


  // Iterate over tick data and set up the phi parameters
  Parameters phi(maxTLAId+1);
  for(list<TickSurv>::const_iterator it = tickdata.begin();
      it != tickdata.end();
      it++)
    {
      char tagbuff[10];
      sprintf(tagbuff, "phi%i", it->region);
      double startval = 0.5;
      phi[it->region] = Parameter(startval, BetaPrior(it->a + it->numpos, it->b + it->total - it->numpos), tagbuff);
    }

  likelihood.SetMovtBan(0.0f);
  likelihood.SetParameters(epsilon, phi, delta, omega, beta1, beta2,
			   nu, alpha1, alpha2, alpha3, zeta, a, b);


  // Set up MCMC algorithm
  Mcmc::Initialize();

  Mcmc::McmcRoot mcmc(likelihood, seed[0]);

  UpdateBlock txDelta;
  txDelta.add(epsilon);
  txDelta.add(beta1);
  txDelta.add(beta2);
  txDelta.add(delta);
  txDelta.add(alpha1);
  txDelta.add(alpha2);
  txDelta.add(alpha3);
  txDelta.add(zeta);
  Mcmc::AdaptiveMultiLogMRW* updateDistance =
      (Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW",
          "txDistance");
  updateDistance->SetParameters(txDelta);

  UpdateBlock updatePhiBlk;
  for(list<TickSurv>::const_iterator it = tickdata.begin();
      it != tickdata.end();
      it++)
    {
      updatePhiBlk.add(phi[it->region]);
    }

  Mcmc::RandomScan* updatePhi = (Mcmc::RandomScan*) mcmc.Create("RandomScan", "updatePhi");
  updatePhi->SetNumReps(10);
  for(list<TickSurv>::const_iterator it = tickdata.begin();
      it != tickdata.end();
      it++)
    {
      UpdateBlock* updBlock = new UpdateBlock;
      updBlock->add(phi[it->region]);
      Mcmc::AdaptiveSingleMRW* upd = (Mcmc::AdaptiveSingleMRW*) updatePhi->Create("AdaptiveSingleMRW",phi[it->region].GetTag());
      upd->SetParameters(*updBlock);
    }

  UpdateBlock infecPeriod;
  infecPeriod.add(a);
  infecPeriod.add(b);
  Mcmc::InfectionTimeUpdate* updateInfecTime =
      (Mcmc::InfectionTimeUpdate*) mcmc.Create("InfectionTimeUpdate",
          "infecTimes");
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
    string outputFile(_outputFile[0]);
    PosteriorHDF5Writer output(outputFile, likelihood);
    output.AddParameter(epsilon);
    output.AddParameter(delta);
    output.AddParameter(omega);
    output.AddParameter(beta1);
    output.AddParameter(beta2);
    output.AddParameter(nu);
    output.AddParameter(alpha1);
    output.AddParameter(alpha2);
    output.AddParameter(alpha3);
    output.AddParameter(zeta);
    output.AddParameter(b);
    for(list<TickSurv>::const_iterator it = tickdata.begin();
	it != tickdata.end();
	it++)
      {
	output.AddParameter(phi[it->region]);
      }

    boost::function< float () > getlikelihood = boost::bind(&GpuLikelihood::GetLogLikelihood, &likelihood);
    output.AddSpecial("loglikelihood",getlikelihood);
    boost::function< float () > getnuminfecs = boost::bind(&GpuLikelihood::GetNumInfecs, &likelihood);
    output.AddSpecial("numInfecs",getnuminfecs);
    boost::function< float () > getmeanI2N = boost::bind(&GpuLikelihood::GetMeanI2N, &likelihood);
    output.AddSpecial("meanI2N", getmeanI2N);
    boost::function< float () > getmeanOccI = boost::bind(&GpuLikelihood::GetMeanOccI, &likelihood);
    output.AddSpecial("meanOccI", getmeanOccI);


    // Run the chain
    for(size_t k=0; k<numIter[0]; ++k)
      {
        if(k % 100 == 0)
          {
	    Rcpp::Rcout << "Iteration " << k << std::endl;
            output.flush();
          }
        mcmc.Update();
        output.write();
      }

    // Wrap up
    map<string, float> acceptance = mcmc.GetAcceptance();

    for(map<string, float>::const_iterator it = acceptance.begin();
        it != acceptance.end(); ++it)
      {
        cout << it->first << ": " << it->second << "\n";
      }

  return outputfile;

}

