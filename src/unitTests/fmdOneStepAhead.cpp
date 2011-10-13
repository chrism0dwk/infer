/*************************************************************************
 *  ./src/unitTests/fmdOneStepAhead.cpp
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
 * fmdOneStepAhead.cpp
 *
 *  Created on: 29 Sep 2011
 *      Author: stsiab
 */


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
#include "OneStepAhead.hpp"
#include "Data.hpp"
#include "PosteriorReader.hpp"

using namespace EpiRisk;

struct Settings
{
  string population;
  string epidemic;
  string posterior;
  string infecTimes;
  string output;
  double obstime;

  void
  load(const string& filename)
  {
    using boost::property_tree::ptree;

    typedef boost::property_tree::ptree_bad_data bad_data;
    typedef boost::property_tree::ptree_bad_path bad_xml;
    typedef boost::property_tree::ptree_error runtime_error;
    ptree pt;

    read_xml(filename, pt);

    population = pt.get<string> ("fmdOneStepAhead.paths.population");
    epidemic = pt.get<string> ("fmdOneStepAhead.paths.epidemic");
    posterior = pt.get<string> ("fmdOneStepAhead.paths.posterior");
    infecTimes = pt.get<string> ("fmdOneStepAhead.paths.infecTimes");
    output = pt.get<string> ("fmdOneStepAhead.paths.output");
    obstime = pt.get<double> ("fmdOneStepAhead.options.obstime");
  }

};



int
main(int argc, char* argv[])
{
  // Simulates from Gillespie sim

  string configFilename;
  int linenum;

  try
    {
      po::options_description desc("Allowed options");
      desc.add_options()("help,h", "Show help message")("config,c", po::value<
          string>(), "config file to use")("linenum,l", po::value<int>(), "posterior line number");

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

      if (vm.count("linenum"))
        {
          linenum = vm["linenum"].as<double> ();
        }
      else
        {
          linenum = -1;
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

  myPopulation.importPopData(*popDataImporter);
  myPopulation.importEpiData(*epiDataImporter);
  myPopulation.setObsTime(settings.obstime);

  delete popDataImporter;
  delete epiDataImporter;

  FmdParameters parameters;
  PosteriorReader posterior(settings.posterior, settings.infecTimes);

  FmdModel model(myPopulation, parameters);
  OneStepAhead<FmdModel>* onestepahead = new OneStepAhead<FmdModel>(model, settings.output);

  int lineCounter = 1;
  bool oneline = false;
  if (linenum != -1) {
      for(int i=0; i<linenum; ++i) posterior.next();
      lineCounter = linenum;
      oneline = true;
  }
  posterior.next();
  do
    {

      std::cout << "Line: " << lineCounter << std::endl;
      std::map<string, double> tmp = posterior.params();
      parameters.gamma1 = tmp["gamma1"];
      parameters.gamma2 = tmp["gamma2"];
      parameters.delta = tmp["delta"];
      parameters.epsilon = tmp["epsilon"];
      parameters.xi_p = tmp["xi_p"];
      parameters.xi_s = tmp["xi_s"];
      parameters.psi_c = tmp["psi_c"];
      parameters.psi_p = tmp["psi_p"];
      parameters.psi_s = tmp["psi_s"];
      parameters.zeta_p = tmp["zeta_p"];
      parameters.zeta_s = tmp["zeta_s"];
      parameters.phi_c = tmp["phi_c"];
      parameters.phi_p = tmp["phi_p"];
      parameters.phi_s = tmp["phi_s"];

      myPopulation.clearInfections();
      for(map<string,double>::const_iterator it = posterior.infecTimes().begin();
          it != posterior.infecTimes().end();
          it++) {
          myPopulation.moveInfectionTime(it->first, it->second);
      }


      onestepahead->compute(lineCounter);
      lineCounter++;
    } while(posterior.next() and !oneline);

  delete onestepahead;

  return EXIT_SUCCESS;
}
