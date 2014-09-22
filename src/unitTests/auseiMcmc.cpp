/*************************************************************************
 *  ./src/unitTests/auseiMcmc.cpp
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

#include "config.h"

#include <iostream>
#include <gsl/gsl_randist.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/exceptions.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "SpatPointPop.hpp"
#include "Mcmc.hpp"
#include "Data.hpp"
#include "McmcWriter.hpp"

using namespace EpiRisk;

///////////////////
// Prior classes //
///////////////////
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
  double a_;
  double b_;
public:
  BetaPrior(const double a, const double b) :
      a_(a), b_(b)
  {
  }
  ;
  double
  operator()(const double x)
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

class GaussianPrior : public Prior
{
  double mu_;
  double var_;
public:
  GaussianPrior(const double mu, const double var) :
      mu_(mu), var_(var)
  {
  }
  ;
  double
  operator()(const double x)
  {
    return gsl_ran_gaussian_pdf(x - mu_, sqrt(var_));
  }
  Prior*
  create() const
  {
    return new GaussianPrior(mu_, var_);
  }
  Prior*
  clone() const
  {
    return new GaussianPrior(*this);
  }
};

//////////////
// Settings //
//////////////
class Settings
{

private:
  std::string popFn_;
  std::string epiFn_;
  std::string outputPfx_;
  double tuneMu_;
  double tuneDelta_;
  double tuneTheta_;
  double tuneB_;
  double obsTime_;
  size_t numIter_;
  size_t seed_;
  Parameters param_;
  double omega_;
  double nu_;

  void
  loadParameters(const boost::property_tree::ptree& pt)
  {
    using namespace boost::property_tree;
    std::vector<std::string> tags(9);
    tags[0] = "epsilon0";
    tags[1] = "mu";
    tags[2] = "epsilon10";
    tags[3] = "xi";
    tags[4] = "zeta";
    tags[5] = "delta";
    tags[6] = "theta";
    tags[7] = "epsilon44";
    tags[8] = "b";

    param_ = Parameters(9);
    try
      {
        cout << "===== Parameters =====" << "\n";
        for (size_t i = 0; i < 9; ++i)
          {
            std::string path = tags[i];
            double init = pt.get<double>(path + ".init");
            std::string dist = pt.get<std::string>(
                path + ".prior.distribution");

            if (dist == "gamma")
              {
                double alpha = pt.get<double>(path + ".prior.alpha");
                double beta = pt.get<double>(path + ".prior.beta");
                param_(i) = Parameter(init, GammaPrior(alpha, beta), tags[i]);
              }
            else if (dist == "beta")
              {
                double alpha = pt.get<double>(path + ".prior.alpha");
                double beta = pt.get<double>(path + ".prior.beta");
                param_(i) = Parameter(init, BetaPrior(alpha, beta), tags[i]);
              }
            else if (dist == "gaussian" or dist == "normal")
              {
                double mean = pt.get<double>(path + ".prior.mean");
                double sd = pt.get<double>(path + ".prior.sd");
                param_(i) = Parameter(init, GaussianPrior(mean, sd), tags[i]);
              }
            else
              {
                std::string msg = "Unknown distribution '" + dist
                    + "' for parameter '" + tags[i] + "'";
                throw EpiRisk::data_exception(msg.c_str());
              }
            cout << param_(i).getTag() << ": " << param_(i) << "\n";
          }
        cout << "======================" << "\n";
      }
    catch (ptree_bad_path& e)
      {
        std::string msg = "Required parameter config '" + std::string(e.what())
            + "' not found";
        throw EpiRisk::data_exception(msg.c_str());
      }
  }

  void
  load(const std::string& filename)
  {
    using boost::property_tree::ptree;
    ptree pt;
    read_xml(filename, pt);
    ptree root = pt.get_child("auseiMcmc");
    popFn_ = root.get<std::string>("path.population");
    epiFn_ = root.get<std::string>("path.epidemic");
    outputPfx_ = root.get<std::string>("path.outputPrefix");
    tuneMu_ = root.get<double>("mcmc.tune.mu");
    tuneDelta_ = root.get<double>("mcmc.tune.delta");
    tuneTheta_ = root.get<double>("mcmc.tune.theta");
    tuneB_ = root.get<double>("mcmc.tune.b");
    numIter_ = root.get<double>("mcmc.iterations");
    seed_ = root.get("mcmc.seed", 0);
    obsTime_ = root.get<double>("options.obstime");
    omega_ = root.get<double>("options.omega");
    nu_ = root.get<double>("options.nu");

    loadParameters(root.get_child("parameters"));
  }

public:
  Settings(const std::string& filename)
  {
    load(filename);
  }

  const std::string&
  getEpiFn() const
  {
    return epiFn_;
  }

  size_t
  getNumIter() const
  {
    return numIter_;
  }

  double
  getObsTime() const
  {
    return obsTime_;
  }

  const std::string&
  getOutputPfx() const
  {
    return outputPfx_;
  }

  const Parameters&
  getParam() const
  {
    return param_;
  }

  const std::string&
  getPopFn() const
  {
    return popFn_;
  }

  size_t
  getSeed() const
  {
    return seed_;
  }

  double
  getTuneB() const
  {
    return tuneB_;
  }

  double
  getTuneDelta() const
  {
    return tuneDelta_;
  }

  double
  getTuneMu() const
  {
    return tuneMu_;
  }

  double
  getTuneTheta() const
  {
    return tuneTheta_;
  }

  double
  getOmega() const
  {
    return omega_;
  }

  double
  getNu() const
  {
    return nu_;
  }

};

////////////////////
// Debug function //
////////////////////
void
dumpData(Population<TestCovars>* popn)
{
  for (Population<TestCovars>::PopulationIterator it = popn->begin();
      it != popn->end(); it++)
    {
      const TestCovars& covars(it->getCovariates());
      cout << it->getId() << "\t" << covars.x << "\t" << covars.y << "\t"
          << covars.horses << "\t" << covars.area << "\t" << covars.vaccdate
          << "\t" << it->getI() << "\t" << it->getN() << "\n";
    }
}

int
main(int argc, char* argv[])
{

//  try
//    {
  // Header
  cout << PACKAGE_STRING << " (c) 2014 " << PACKAGE_BUGREPORT << "\n\n";
  // Parse command line options
  size_t numIter, seed;
  std::string outputPfx, configFile;
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "display help message")("config,c",
      po::value<std::string>(&configFile), "xml config file (required)")(
      "niter,n", po::value<size_t>(&numIter)->default_value(10000),
      "number of iterations")("seed,s",
      po::value<size_t>(&seed)->default_value(0), "random seed")("output,o",
      po::value<std::string>(&outputPfx)->default_value(std::string("ausei")),
      "output file prefix");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (!vm.count("config") or vm.count("help"))
    {
      cout << desc << "\n";
      return EXIT_FAILURE;
    }

  // Read config file
  Settings settings(configFile);
  if (vm.find("niter,n") == vm.end())
    numIter = settings.getNumIter();
  if (vm.find("seed,s") == vm.end())
    seed = settings.getSeed();
  if (vm.find("output,o") == vm.end())
    outputPfx = settings.getOutputPfx();

  // Set up MPI communicators
  mpi::environment env(argc, argv);
  mpi::communicator comm;

  // Construct the population
  typedef Population<TestCovars> MyPopulation;

  PopDataImporter* popDataImporter = new PopDataImporter(settings.getPopFn());
  EpiDataImporter* epiDataImporter = new EpiDataImporter(settings.getEpiFn());
  Population<TestCovars>* myPopulation = new Population<TestCovars>;

  myPopulation->importPopData(*popDataImporter);
  myPopulation->importEpiData(*epiDataImporter);
  myPopulation->setObsTime(settings.getObsTime());

  delete popDataImporter;
  delete epiDataImporter;

  cout << "Population size: " << myPopulation->size() << endl;
  cout << "Num infectives: " << myPopulation->numInfected() << endl;

  // Set initial value, prior, and tag for each parameter
  Parameters txparams = settings.getParam();

  // Unused
  Parameters dxparams(1);
  dxparams(0) = Parameter(0.01, GammaPrior(1, 1), "null");
  // /Unused

  // Construct MCMC algorithm
  Mcmc* myMcmc = new Mcmc(*myPopulation, txparams, dxparams, seed);

  myMcmc->newSingleSiteLogMRW(txparams(1), settings.getTuneMu());
  myMcmc->newSingleSiteLogMRW(txparams(5), settings.getTuneDelta());
  myMcmc->newSingleSiteMRW(txparams(6), settings.getTuneTheta());
  myMcmc->newWithinFarmBetaLogMRW(txparams(8), settings.getOmega(),
      settings.getNu(), settings.getTuneB());

  UpdateBlock background;
  background.add(txparams(0));
  background.add(txparams(2));
  background.add(txparams(7));
  myMcmc->newAdaptiveMultiLogMRW("background", background, 500);

  UpdateBlock area;
  area.add(txparams(3));
  area.add(txparams(4));
  myMcmc->newAdaptiveMultiMRW("area", area, 500);

  std::string parmFn(outputPfx + ".parms");
  std::string occFn(outputPfx + ".occ");

  McmcWriter<MyPopulation>* writer = new McmcWriter<MyPopulation>(parmFn,
      occFn);

  // Run the MCMC
  map<string, double> acceptance = myMcmc->run(numIter, *writer);

  delete myMcmc;
  delete writer;
  delete myPopulation;
//    }
//  catch (std::exception& e)
//    {
//      cerr << "Exception occurred: " << e.what() << endl;
//      return EXIT_FAILURE;
//    }

  return EXIT_SUCCESS;

}
