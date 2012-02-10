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
#include <gsl/gsl_randist.h>
#include <sstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

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
  myPopulation->setObsTime(325.0);

  delete popDataImporter;
  delete epiDataImporter;


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

  delete myMcmc;
  delete myPopulation;

  return EXIT_SUCCESS;

}
