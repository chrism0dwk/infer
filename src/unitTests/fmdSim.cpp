/*************************************************************************
 *  ./src/unitTests/fmdSim.cpp
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

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "fmdModel.hpp"
#include "GillespieSim.hpp"
#include "Data.hpp"
#include "Random.hpp"

using namespace EpiRisk;

struct Settings
{
  string population;
  string epidemic;
  string output;

  double minTime, maxTime;
  bool simCensoredEvents;
  double ntor;

  double epsilon;
  double gamma1;
  double gamma2;
  double delta;
  double xi_p, xi_s;
  double psi_c, psi_p, psi_s;
  double zeta_p, zeta_s;
  double phi_c, phi_p, phi_s;
  double a, b;

  void
  load(const string& filename)
  {
    using boost::property_tree::ptree;

    typedef boost::property_tree::ptree_bad_data bad_data;
    typedef boost::property_tree::ptree_bad_path bad_xml;
    typedef boost::property_tree::ptree_error runtime_error;
    ptree pt;

    read_xml(filename, pt);

    population = pt.get<string> ("fmdGillespieSim.paths.population");
    epidemic = pt.get<string> ("fmdGillespieSim.paths.epidemic");
    output = pt.get<string> ("fmdGillespieSim.paths.output");

    minTime = pt.get<double> ("fmdGillespieSim.options.mintime", 0);
    maxTime = pt.get<double> ("fmdGillespieSim.options.maxtime", (double)POSINF);
    simCensoredEvents = pt.get<bool> ("fmdGillespieSim.options.simcensoredevents", true);
    ntor = pt.get<double> ("fmdGillespieSim.constants.ntor");

    epsilon = pt.get<double> ("fmdGillespieSim.parameters.epsilon");
    gamma1 = pt.get<double> ("fmdGillespieSim.parameters.gamma1");
    gamma2 = pt.get<double> ("fmdGillespieSim.parameters.gamma2");
    delta = pt.get<double> ("fmdGillespieSim.parameters.delta");
    xi_p = pt.get<double> ("fmdGillespieSim.parameters.xi_p");
    xi_s = pt.get<double> ("fmdGillespieSim.parameters.xi_s");
    psi_c = pt.get<double> ("fmdGillespieSim.parameters.psi_c");
    psi_p = pt.get<double> ("fmdGillespieSim.parameters.psi_p");
    psi_s = pt.get<double> ("fmdGillespieSim.parameters.psi_s");
    zeta_p = pt.get<double> ("fmdGillespieSim.parameters.zeta_p");
    zeta_s = pt.get<double> ("fmdGillespieSim.parameters.zeta_s");
    phi_c = pt.get<double> ("fmdGillespieSim.parameters.phi_c");
    phi_p = pt.get<double> ("fmdGillespieSim.parameters.phi_p");
    phi_s = pt.get<double> ("fmdGillespieSim.parameters.phi_s");
    a = pt.get<double> ("fmdGillespieSim.parameters.a");
    b = pt.get<double> ("fmdGillespieSim.parameters.b");
  }

};

int
main(int argc, char* argv[])
{
  // Simulates from Gillespie sim

  string configFilename;
  int seed;

  try
    {
      po::options_description desc("Allowed options");
      desc.add_options()("help,h", "Show help message")("config,c", po::value<
          string>(), "config file to use")("seed,s", po::value<int>(),
          "random seed (default 0)");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help"))
        {
          cout << desc << "\n";
          return EXIT_FAILURE;
        }

      if (vm.count("config"))
        {
          configFilename = vm["config"].as<string> ();
        }
      else
        {
          cerr << "Config file required" << "\n";
          cerr << desc << "\n";
          return EXIT_FAILURE;
        }

      if (vm.count("seed"))
        {
          seed = vm["seed"].as<int> ();
        }
      else
        {
          seed = 0;
        }
    }
  catch (exception& e)
    {
      cerr << "Exception: " << e.what() << "\n";
      return 2;
    }
  catch (...)
    {
      cerr << "Unknown exception" << "\n";
      return 2;
    }

  // Read in config file
  Settings settings;
  try
    {
      settings.load(configFilename);
    }
  catch (exception& e)
    {
      cerr << "Loading config failed: " << e.what() << endl;
      return EXIT_FAILURE;
    }

  PopDataImporter* popDataImporter = new PopDataImporter(settings.population);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(settings.epidemic);

  FmdModel::PopulationType myPopulation;

  myPopulation.setObsTime(settings.minTime);
  myPopulation.importPopData(*popDataImporter);
  myPopulation.importEpiData(*epiDataImporter);


  delete popDataImporter;
  delete epiDataImporter;

  FmdParameters parameters;
  parameters.gamma1 = Parameter(settings.gamma1, GammaPrior(1, 1), "gamma1");
  parameters.gamma2 = Parameter(settings.gamma2, GammaPrior(1, 1), "gamma2");
  parameters.delta = Parameter(settings.delta, GammaPrior(1, 1), "delta");
  parameters.epsilon = Parameter(settings.epsilon, GammaPrior(1, 1), "epsilon");
  parameters.xi_p = Parameter(settings.xi_p, GammaPrior(1, 1), "xi_p");
  parameters.xi_s = Parameter(settings.xi_s, GammaPrior(1, 1), "xi_s");
  parameters.psi_c = Parameter(settings.psi_c, BetaPrior(2, 2), "psi_c");
  parameters.psi_p = Parameter(settings.psi_p, BetaPrior(2, 2), "psi_p");
  parameters.psi_s = Parameter(settings.psi_s, BetaPrior(2, 2), "psi_s");
  parameters.zeta_p = Parameter(settings.zeta_p, GammaPrior(1, 1), "zeta_p");
  parameters.zeta_s = Parameter(settings.zeta_s, GammaPrior(1, 1), "zeta_s");
  parameters.phi_c = Parameter(settings.phi_c, BetaPrior(2, 2), "phi_c");
  parameters.phi_p = Parameter(settings.phi_p, BetaPrior(2, 2), "phi_p");
  parameters.phi_s = Parameter(settings.phi_s, BetaPrior(2, 2), "phi_s");
  parameters.a = Parameter(settings.a, GammaPrior(1, 1), "a");
  parameters.b = Parameter(settings.b, GammaPrior(1, 1), "b");

  FmdModel model(myPopulation, parameters);
  Random random(seed);

  GillespieSim<FmdModel> simulation(model, random);
  simulation.setMaxTime(settings.maxTime);

  simulation.simulate(settings.simCensoredEvents);

  stringstream s(settings.output);
  s << "." << seed;
  simulation.serialize(settings.output);
  return EXIT_SUCCESS;
}
