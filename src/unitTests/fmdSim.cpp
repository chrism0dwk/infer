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

#include "fmdModel.hpp"
#include "GillespieSim.hpp"
#include "Data.hpp"
#include "Random.hpp"

using namespace EpiRisk;

int main(int argc, char* argv[])
{
  // Simulates from Gillespie sim
  if (argc != 5) {
      std::cerr << "Usage: fmdSim <pop file> <epi file> <output prefix> <num sims>\n";
      return EXIT_FAILURE;
  }

  PopDataImporter* popDataImporter = new PopDataImporter(argv[1]);
  EpiDataImporter* epiDataImporter = new EpiDataImporter(argv[2]);

  FmdModel::PopulationType myPopulation;

  myPopulation.importPopData(*popDataImporter);
  myPopulation.importEpiData(*epiDataImporter);
  myPopulation.setObsTime(18.0);

  delete popDataImporter;
  delete epiDataImporter;

  FmdParameters parameters;
  parameters.gamma1 = Parameter(0.65,GammaPrior(1,1),"gamma1");
  parameters.gamma2 = Parameter(1.0,GammaPrior(1,1),"gamma2");
  parameters.delta = Parameter(1.1,GammaPrior(1,1),"delta");
  parameters.epsilon = Parameter(3.5,GammaPrior(1,1),"epsilon");
  parameters.xi_p = Parameter(0.18,GammaPrior(1,1),"xi_p");
  parameters.xi_s = Parameter(0.13,GammaPrior(1,1),"xi_s");
  parameters.psi_c = Parameter(1,BetaPrior(2,2),"psi_c");
  parameters.psi_p = Parameter(1,BetaPrior(2,2),"psi_p");
  parameters.psi_s = Parameter(1,BetaPrior(2,2),"psi_s");
  parameters.zeta_p = Parameter(0.14,GammaPrior(1,1),"zeta_p");
  parameters.zeta_s = Parameter(0.2,GammaPrior(1,1),"zeta_s");
  parameters.phi_c = Parameter(1,BetaPrior(2,2),"phi_c");
  parameters.phi_p = Parameter(1,BetaPrior(2,2),"phi_p");
  parameters.phi_s = Parameter(1,BetaPrior(2,2),"phi_s");
  parameters.a = Parameter(0.015,GammaPrior(1,1),"a");
  parameters.b = Parameter(0.8,GammaPrior(1,1),"b");


  FmdModel model(myPopulation,parameters);
  Random random(0);

  GillespieSim<FmdModel> simulation(model, random);

  simulation.simulate();

  simulation.serialize("testFile.txt");

}
