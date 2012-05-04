/*************************************************************************
 *  ./src/mcmc/Mcmc.hpp
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

/* Header file for aifuncs.cpp */

#ifndef INCLUDE_MCMC_H
#define INCLUDE_MCMC_H

#include <math.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include <vector>
#include <map>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/ptr_container/ptr_list.hpp>



#include "types.hpp"
#include "GpuLikelihood.hpp"
#include "Parameter.hpp"
#include "Random.hpp"
#include "EmpCovar.hpp"
#include "McmcWriter.hpp"
#include "MCMCUpdater.hpp"
#include "McmcLikelihood.hpp"


namespace EpiRisk
{
  using namespace std;
  using namespace EpiRisk;

  // FWD DECLS
  class McmcUpdate;
  class SingleSiteLogMRW;
  class AdaptiveMultiLogMRW;
  class AdaptiveMultiMRW;
  class SpeciesMRW;
  class SusceptibilityMRW;
  class InfectivityMRW;
  class InfectionTimeGammaCentred;
  class InfectionTimeGammaNC;
  class InfectionTimeUpdate;
  class SellkeSerializer;




  class Mcmc
  {

    McmcLikelihood likelihood_;
    Random* random_;

    boost::ptr_list<McmcUpdate> updateStack_;

    std::vector<size_t> elements_;
    ofstream mcmcOutput_;
    ofstream stdout_;

    double timeCalc_,timeUpdate_;

    // Likelihood functions
    float Propose();
    void AcceptProposal();
    void RejectProposal();

    // Infection time functions -- should really be an updater
    bool
    UpdateI();
    bool
    AddI();
    bool
    DeleteI();

    void
    DumpParms() const;

  public:
    Mcmc(GpuLikelihood& logLikelihood, const size_t randomSeed);
    ~Mcmc();
    void
    Update();
    //! Creates a single site log MRW updater
    SingleSiteLogMRW*
    NewSingleSiteLogMRW(Parameter& param, const double tuning);
    //! Creates a block update group
    AdaptiveMultiLogMRW*
    NewAdaptiveMultiLogMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    AdaptiveMultiMRW*
    NewAdaptiveMultiMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    SpeciesMRW*
    NewSpeciesMRW(const string tag, UpdateBlock& params, std::vector<double>& alpha, const size_t burnin = 1000);
    InfectivityMRW*
    NewInfectivityMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    SusceptibilityMRW*
    NewSusceptibilityMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    InfectionTimeGammaCentred*
    NewInfectionTimeGammaCentred(const string tag, Parameter& param, const float tuning);
    InfectionTimeGammaNC*
    NewInfectionTimeGammaNC(const string tag, Parameter& param, const float tuning, const float ncProp);
    InfectionTimeUpdate*
    NewInfectionTimeUpdate(const string tag, Parameter& a, Parameter& b, const size_t reps);
    SellkeSerializer*
    NewSellkeSerializer(const string filename);

    std::map<std::string, float>
    GetAcceptance() const;
    void
    ResetAcceptance();


  };

}
#endif
