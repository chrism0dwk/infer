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
#include <fstream>
#include <gsl/gsl_randist.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "config.h"
#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"

#define CONNECTIONCUTOFF 25.0


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

  struct ConnectionPredicate
  {
    bool operator() (Population<TestCovars>::Individual i, Population<TestCovars>::Individual j)
    {
      double dx = i.getCovariates().x - j.getCovariates().x;
      double dy = i.getCovariates().y - j.getCovariates().y;

      if(dx > CONNECTIONCUTOFF or dy > CONNECTIONCUTOFF) return false;
      else return sqrt(dx*dx + dy*dy) <= CONNECTIONCUTOFF;
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
  ParamSetting xi_p,xi_s;
  ParamSetting psi_c,psi_p,psi_s;
  ParamSetting zeta_p,zeta_s;
  ParamSetting phi_c,phi_p,phi_s;
  ParamSetting a,b;
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

      read_xml(filename,pt);

      populationfile = pt.get<string> ("fmdMcmc.paths.population");
      epidemicfile = pt.get<string> ("fmdMcmc.paths.epidemic");
      connectionfile = pt.get<string> ("fmdMcmc.paths.connections");
      posteriorfile = pt.get<string> ("fmdMcmc.paths.posterior");

      obstime = pt.get<double> ("fmdMcmc.options.obstime",POSINF);
      iterations = pt.get<double> ("fmdMcmc.options.iterations",1);
      iupdates = pt.get<double> ("fmdMcmc.options.iupdates",0);
      seed = pt.get<int> ("fmdMcmc.options.seed",1);

      parameters.epsilon.value = pt.get<double> ("fmdMcmc.parameters.epsilon.value",0.5);
      parameters.epsilon.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.epsilon.prior.gamma.a",1);
      parameters.epsilon.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.epsilon.prior.gamma.b",1);

      parameters.gamma1.value = pt.get<double> ("fmdMcmc.parameters.gamma1.value",0.5);
      parameters.gamma1.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.gamma1.prior.gamma.a",1);
      parameters.gamma1.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.gamma1.prior.gamma.b",1);

      parameters.gamma2.value = pt.get<double> ("fmdMcmc.parameters.gamma2.value",0.5);
      parameters.gamma2.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.gamma2.prior.gamma.a",1);
      parameters.gamma2.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.gamma2.prior.gamma.b",1);

      parameters.delta.value = pt.get<double> ("fmdMcmc.parameters.delta.value",0.5);
      parameters.delta.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.delta.prior.gamma.a",1);
      parameters.delta.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.delta.prior.gamma.b",1);

      parameters.xi_p.value = pt.get<double> ("fmdMcmc.parameters.xi_p.value",0.5);
      parameters.xi_p.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.xi_p.prior.gamma.a",1);
      parameters.xi_p.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.xi_p.prior.gamma.b",1);

      parameters.xi_s.value = pt.get<double> ("fmdMcmc.parameters.xi_s.value",0.5);
      parameters.xi_s.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.xi_s.prior.gamma.a",1);
      parameters.xi_s.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.xi_s.prior.gamma.b",1);

      parameters.phi_c.value = pt.get<double> ("fmdMcmc.parameters.phi_c.value",0.5);
      parameters.phi_c.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.phi_c.prior.beta.a",1);
      parameters.phi_c.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.phi_c.prior.beta.b",1);

      parameters.phi_p.value = pt.get<double> ("fmdMcmc.parameters.phi_p.value",0.5);
      parameters.phi_p.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.phi_p.prior.beta.a",1);
      parameters.phi_p.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.phi_p.prior.beta.b",1);

      parameters.phi_s.value = pt.get<double> ("fmdMcmc.parameters.phi_s.value",0.5);
      parameters.phi_s.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.phi_s.prior.beta.a",1);
      parameters.phi_s.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.phi_s.prior.beta.b",1);

      parameters.zeta_p.value = pt.get<double> ("fmdMcmc.parameters.zeta_p.value",0.5);
      parameters.zeta_p.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.zeta_p.prior.gamma.a",1);
      parameters.zeta_p.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.zeta_p.prior.gamma.b",1);

      parameters.zeta_s.value = pt.get<double> ("fmdMcmc.parameters.zeta_s.value",0.5);
      parameters.zeta_s.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.zeta_s.prior.gamma.a",1);
      parameters.zeta_s.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.zeta_s.prior.gamma.b",1);

      parameters.psi_c.value = pt.get<double> ("fmdMcmc.parameters.psi_c.value",0.5);
      parameters.psi_c.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.psi_c.prior.beta.a",1);
      parameters.psi_c.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.psi_c.prior.beta.b",1);

      parameters.psi_p.value = pt.get<double> ("fmdMcmc.parameters.psi_p.value",0.5);
      parameters.psi_p.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.psi_p.prior.beta.a",1);
      parameters.psi_p.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.psi_p.prior.beta.b",1);

      parameters.psi_s.value = pt.get<double> ("fmdMcmc.parameters.psi_s.value",0.5);
      parameters.psi_s.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.psi_s.prior.beta.a",1);
      parameters.psi_s.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.psi_s.prior.beta.b",1);

      parameters.a.value = pt.get<double> ("fmdMcmc.parameter.a.value",0.08);
      parameters.b.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.a.prior.gamma.a",1);
      parameters.b.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.a.prior.gamma.b",1);

      parameters.b.value = pt.get<double> ("fmdMcmc.parameter.b.value",0.005);
      parameters.b.priorparams[0] = pt.get<double> ("fmdMcmc.parameters.b.prior.gamma.a",1);
      parameters.b.priorparams[1] = pt.get<double> ("fmdMcmc.parameters.b.prior.gamma.b",1);

    }
};



int main(int argc, char* argv[])
{
  // Tests out class Mcmc

  mpi::environment env(argc,argv);
  mpi::communicator comm;

  cerr << PACKAGE_NAME << " " << PACKAGE_VERSION <<  " compiled " << __DATE__ << " " << __TIME__ << endl;

  if (argc != 6) {
      cerr << "Usage: testSpatPointPop <pop file> <epi file> <output folder> <obs time> <num iterations>" << endl;
      return EXIT_FAILURE;
  }

  typedef Population<TestCovars> MyPopulation;

  // Make output directory
  string outputFolder(argv[3]);
  mkdir(outputFolder.c_str(),S_IFDIR | S_IRWXU);

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  //myPopulation->createConnectionGraph(ConnectionPredicate());
  myPopulation->loadConnectionGraph("/storage/stsiab/FMD2001/data/fmd2001_uk_15km.con");
  myPopulation->setObsTime(atof(argv[4]));

  delete popDataImporter;
  delete epiDataImporter;

//  ofstream confile;
//  confile.open("/storage/stsiab/FMD2001/data/fmd2001_uk_15km.con",ios::out);
//  size_t counter = 0;
//  size_t numcons = 0;
//  for(Population<TestCovars>::PopulationIterator it = myPopulation->begin();
//      it != myPopulation->end();
//      it++) {
//
//    for(Population<TestCovars>::PopulationIterator jt = myPopulation->begin();
//        jt != it;
//        jt++ )
//      {
//        double dx = it->getCovariates().x - jt->getCovariates().x;
//        double dy = it->getCovariates().y - jt->getCovariates().y;
//        if(dx <= 15.0 and dy <= 15.0) {
//            if(sqrt(dx*dx + dy*dy) <= 15.0) {
//                confile << it->getId() << "," << jt->getId() << endl;
//                confile << jt->getId() << "," << it->getId() << endl;
//            }
//        }
//      }
//    cout << counter << ": " << numcons << endl;
//    counter++;
//  }
//  confile.close();
//  return 0;

  // Data covariance matrix
//  EmpCovar<LogTransform>::CovMatrix speciesCovar;
//  ifstream covMatrix;
//  covMatrix.open(argv[3],ios::in);
//  covMatrix >> speciesCovar;
//  covMatrix.close();


  Parameters txparams(19);
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
  txparams(18) = Parameter(0.0,GammaPrior(1,1),"logLikelihood");

  Parameters dxparams(1);
  dxparams(0) = Parameter(0.1,GammaPrior(1,1),"null");

  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams,0);
  myMcmc->setNumIUpdates(0);

  std::vector<double> infAlpha(3);
  infAlpha[0] = 757.34;
  infAlpha[1] = 633.37;
  infAlpha[2] = 87.0;

  std::vector<double> suscAlpha(3);
  std::fill(suscAlpha.begin(),suscAlpha.end(),7936);

//  myMcmc->newSingleSiteLogMRW(txparams[0],0.3);
//  myMcmc->newSingleSiteLogMRW(txparams[3],0.8);

  UpdateBlock txDelta;
  txDelta.add(txparams[0]);
  txDelta.add(txparams[1]);
  txDelta.add(txparams[2]);
  txDelta.add(txparams[3]);
  AdaptiveMultiLogMRW* updateDistance = myMcmc->newAdaptiveMultiLogMRW("txDistance",txDelta, 1000);


  UpdateBlock txPsi;
  txPsi.add(txparams[7]);
  txPsi.add(txparams[8]);
  txPsi.add(txparams[9]);
  AdaptiveMultiLogMRW* updatePsi = myMcmc->newAdaptiveMultiLogMRW("txPsi",txPsi, 1000);


  UpdateBlock txPhi;
  txPhi.add(txparams[13]);
  txPhi.add(txparams[14]);
  txPhi.add(txparams[15]);
  AdaptiveMultiLogMRW* updatePhi = myMcmc->newAdaptiveMultiLogMRW("txPhi",txPhi, 1000);

  UpdateBlock txInfec;
  txInfec.add(txparams[0]);
  txInfec.add(txparams[5]);
  txInfec.add(txparams[6]);
  InfectivityMRW* updateInfec = myMcmc->newInfectivityMRW("txInfec",txInfec, txPsi, 1000);

  UpdateBlock txSuscep;
  txSuscep.add(txparams[0]);
  txSuscep.add(txparams[11]);
  txSuscep.add(txparams[12]);
  SusceptibilityMRW* updateSuscep = myMcmc->newSusceptibilityMRW("txSuscep",txSuscep, txPhi, 1000);

//  AdaptiveMultiMRW* updatePhiLin = myMcmc->newAdaptiveMultiMRW("txPhiLin",txPhi,1000);
//  AdaptiveMultiMRW* updatePsiLin = myMcmc->newAdaptiveMultiMRW("txPsiLin",txPsi, 1000);
  AdaptiveMultiMRW* updateDistanceLin = myMcmc->newAdaptiveMultiMRW("txDistanceLin",txDelta, 1000);

  //SellkeSerializer* sellke = myMcmc->newSellkeSerializer(outputFolder + "/sellke.asc");


  stringstream parmFn;
  stringstream occFn;
  stringstream covFn;


  parmFn << outputFolder << "/parameters.asc";
  occFn << outputFolder << "/infec.asc";
  covFn << outputFolder << "/covariances.asc";

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn.str(),occFn.str());

  size_t numIters;
  stringstream iters(argv[5]);
  iters >> numIters;

  map<string,double> acceptance = myMcmc->run(numIters, *writer);

  ofstream covFile(covFn.str().c_str());
  if(covFile.is_open()) {
      covFile << "txDistance:" << updateDistance->getCovariance() << "\n";
      covFile << "txPsi:" << updatePsi->getCovariance() << "\n";
      covFile << "txPhi:" << updatePhi->getCovariance() << "\n";
      covFile << "txInfec:" << updateInfec->getCovariance() << "\n";
      covFile << "txSuscep:" << updateSuscep->getCovariance() << "\n";
  }
  covFile.close();

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
