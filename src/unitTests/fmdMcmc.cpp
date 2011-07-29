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
#include <gsl/gsl_randist.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <sstream>

#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"


using namespace EpiRisk;

  class GammaPrior : public Prior
  {
    double shape_;
    double rate_;
  public:
    GammaPrior(const double shape, const double rate)
    {
      shape_ = shape;
      rate_ = rate;
    }
    double
    operator()(const double x)
    {
      return gsl_ran_gamma_pdf(x,shape_,1/rate_);
    }
    Prior*
    create() const
    {
      return new GammaPrior(shape_,rate_);
    }
    Prior*
    clone() const
    {
      return new GammaPrior(*this);
    }
  };

  class BetaPrior : public Prior
  {
    double a_;
    double b_;
  public:
    BetaPrior(const double a, const double b) : a_(a),b_(b) {};
    double operator()(const double x)
    {
      return gsl_ran_beta_pdf(x,a_,b_);
    }
    Prior*
    create() const
    {
      return new BetaPrior(a_,b_);
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
    InfSuscSN(Parameter& A, Parameter& B) : A_(&A), StochasticNode(B)
    {
    }
    InfSuscSN*
    clone()
    {
      return new InfSuscSN(*this);
    }
    double getValue() const
    {
      return *A_ * *param_;
    }
    void setValue(const double x)
    {
      *param_ = x / *A_;
    }
  };



int main(int argc, char* argv[])
{
  // Tests out class Mcmc

  mpi::environment env(argc,argv);
  mpi::communicator comm;

  if (argc != 5) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <covar matrix> <num iterations>" << endl;
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

  // Data covariance matrix
  EmpCovar<LogTransform>::CovMatrix speciesCovar;
  ifstream covMatrix;
  covMatrix.open(argv[3],ios::in);
  covMatrix >> speciesCovar;
  covMatrix.close();


  Parameters txparams(18);
  txparams(0) = Parameter(0.0108081,GammaPrior(1,1),"gamma1");
  txparams(1) = Parameter(0.5,GammaPrior(1,1),"gamma2");
  txparams(2) = Parameter(1.14985,GammaPrior(1,1),"delta");
  txparams(3) = Parameter(7.72081e-05,GammaPrior(1,1),"epsilon");
  txparams(4) = Parameter(1.0,GammaPrior(1,1),"xi_c");
  txparams(5) = Parameter(0.00205606,GammaPrior(1,1),"xi_p");
  txparams(6) = Parameter(0.613016,GammaPrior(1,1),"xi_s");
  txparams(7) = Parameter(0.237344,BetaPrior(2,2),"psi_c");
  txparams(8) = Parameter(0.665464,BetaPrior(2,2),"psi_p");
  txparams(9) = Parameter(0.129998,BetaPrior(2,2),"psi_s");
  txparams(10) = Parameter(1.0,GammaPrior(1,1),"zeta_c");
  txparams(11) = Parameter(0.000295018,GammaPrior(1,1),"zeta_p");
  txparams(12) = Parameter(0.259683,GammaPrior(1,1),"zeta_s");
  txparams(13) = Parameter(0.402155,BetaPrior(2,2),"phi_c");
  txparams(14) = Parameter(0.749019,BetaPrior(2,2),"phi_p");
  txparams(15) = Parameter(0.365774,BetaPrior(2,2),"phi_s");
  txparams(16) = Parameter(0.0,GammaPrior(1,1),"meanI2N");
  txparams(17) = Parameter(0.0,GammaPrior(1,1),"meanOccI");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,0);
  std::vector<double> infAlpha(3);
  infAlpha[0] = 757.34;
  infAlpha[1] = 633.37;
  infAlpha[2] = 87.0;

  std::vector<double> suscAlpha(3);
  std::fill(suscAlpha.begin(),suscAlpha.end(),7936);

  myMcmc->newSingleSiteLogMRW(txparams[0],0.1);
  myMcmc->newSingleSiteLogMRW(txparams[3],0.1);

  UpdateBlock txInfec;
  txInfec.add(txparams[0]);
  txInfec.add(txparams[5]);
  txInfec.add(txparams[6]);
  myMcmc->newSpeciesMRW("txInfec",txInfec, infAlpha);

  UpdateBlock txSuscep;
  txSuscep.add(txparams[0]);
  txSuscep.add(txparams[11]);
  txSuscep.add(txparams[12]);
  myMcmc->newSpeciesMRW("txSuscep",txSuscep, suscAlpha);

  UpdateBlock txDelta;
  txDelta.add(txparams[0]);
  txDelta.add(txparams[1]);
  txDelta.add(txparams[2]);
  txDelta.add(txparams[3]);
  AdaptiveMultiLogMRW* tx = myMcmc->newAdaptiveMultiLogMRW("txDistance",txDelta, 1000);

  UpdateBlock txPsi;
  txPsi.add(txparams[7]);
  txPsi.add(txparams[8]);
  txPsi.add(txparams[9]);
  myMcmc->newAdaptiveMultiLogMRW("txPsi",txPsi, 1000);

  UpdateBlock txPhi;
  txPhi.add(txparams[13]);
  txPhi.add(txparams[14]);
  txPhi.add(txparams[15]);
  myMcmc->newAdaptiveMultiLogMRW("txPhi",txPhi, 1000);

  stringstream parmFn;
  stringstream occFn;

  parmFn << "/scratch/stsiab/FMD2001/output/fmdTestSim.trunc100.1.p" << comm.size() << ".parms";
  occFn << "/scratch/stsiab/FMD2001/output/fmdTestSim.trunc100.1.p" << comm.size() << ".occ";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[4]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  if(comm.rank() == 0) {
      cout << "Infection acceptance: " << acceptance["I"] << endl;
      cout << "Addition acceptance: " << acceptance["add"] << endl;
      cout << "Deletion acceptance: " << acceptance["delete"] << endl;
  }

  delete myMcmc;
  delete writer;
  delete myPopulation;

  return EXIT_SUCCESS;

}
