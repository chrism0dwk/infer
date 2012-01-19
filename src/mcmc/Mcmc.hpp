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
#include <amdlibm.h>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/ptr_container/ptr_list.hpp>



#include "types.hpp"
#include "SpatPointPop.hpp"
#include "Data.hpp"
#include "Parameter.hpp"
#include "Random.hpp"
#include "EmpCovar.hpp"
#include "McmcWriter.hpp"
#include "MCMCUpdater.hpp"



namespace EpiRisk
{
  using namespace std;
  using namespace EpiRisk;
  namespace mpi = boost::mpi;

  // FWD DECLS
  class McmcUpdate;
  class SingleSiteLogMRW;
  class AdaptiveMultiLogMRW;
  class AdaptiveMultiMRW;
  class SpeciesMRW;
  class SusceptibilityMRW;
  class InfectivityMRW;
  class SusceptibilityPowMRW;
  class InfectivityPowMRW;
  class SellkeSerializer;

  struct DIC {
     double Dbar;
     double Dhat;
     double pD;
     double DIC;
  };

  struct Likelihood
  {
    double local;
    double global;
    map<string, double> productCache;
    map<string, double> integPressure;
  };

  class Mcmc
  {

    //// Update algorithms ////
//    friend class McmcUpdater;
//    friend class AdaptiveMultiLogMRW;

    friend class SusceptibilityMRW;
    friend class InfectivityMRW;
    friend class SusceptibilityPowMRW;
    friend class InfectivityPowMRW;
    friend class SellkeSerializer;

    Population<TestCovars>& pop_;
    Parameters& txparams_;
    Parameters& dxparams_;
    Likelihood logLikelihood_;
    Likelihood testLik_;
    Random* random_;

    boost::ptr_list<McmcUpdate> updateStack_;
    mpi::communicator comm_;
    int mpirank_, mpiprocs_;
    bool mpiInitHere_;
    bool accept_;
    double integPressTime_;
    std::vector<size_t> elements_;
    ofstream mcmcOutput_;
    ofstream stdout_;

    struct
    ProcessInfectivesCmp
    {
      bool
      operator()(const Population<TestCovars>::InfectiveIterator& lhs, const Population<TestCovars>::InfectiveIterator& rhs) const
      {
        return lhs->getId() < rhs->getId();
      }
    };

    typedef set<Population<TestCovars>::InfectiveIterator, ProcessInfectivesCmp> ProcessInfectives;
    ProcessInfectives processInfectives_;
    ProcessInfectives occultList_;
    ProcessInfectives dcList_;

    //// THESE SHOULD BE IN A "MODEL" CLASS ////
    virtual
    double
    susceptibility(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const;
    virtual
    double
    infectivity(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const;
    double
    infecsuscep(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const;
    virtual
    double
    beta(const Population<TestCovars>::Individual& i, const Population<
        TestCovars>::Individual& j) const;
    virtual
    double
    betastar(const Population<TestCovars>::Individual& i, const Population<
        TestCovars>::Individual& j) const;
    double
    instantPressureOn(const Population<TestCovars>::InfectiveIterator& j,
        const double Ij);
    double
    integPressureOn(const Population<TestCovars>::PopulationIterator& j,
        const double Ij);

    void
    updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
        const double newTime, Likelihood& updatedLogLik);

    void
    newUpdateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j,
        const double newTime, Likelihood& updatedLogLik);
    /////////////////////////////////////////////


    bool
    updateI(const size_t index = 0);
    bool
    addI();
    bool
    deleteI();
    void
        moveProdCache(const string id, const size_t fromIndex,
            const size_t toIndex);

    /// DIC Methods -- again, should be a separate class
    DIC dic_;
    size_t dicUpdates_;
    double postMeanDev_;
    map<string,double> meanInfecTimes_;
    Parameters meanParams_;
    size_t numIUpdates_;
    double timeCalc_,timeUpdate_;
    size_t numCalc_,numUpdate_;

    void
    initializeDIC();
    void
    updateDIC();

    double
    getMeanI2N() const;

    double
    getMeanOccI() const;

    void
    dumpParms() const;
    void
    dumpProdCache();
    void
    checkProcPopConsistency();
    void
    checkInfecOrder();
    void
    loadBalance();

  public:
    Mcmc(Population<TestCovars>& population, Parameters& transParams,
        Parameters& detectParams, const size_t randomSeed);
    ~Mcmc();
    //    void
    //    pushUpdater(const string tag, const ParameterGroup& updateGroup);
    double
    getLogLikelihood() const;
    map<string, double>
        run(const size_t numIterations,
            McmcWriter<Population<TestCovars> >& writer);
    //! Creates a single site log MRW updater
    SingleSiteLogMRW*
    newSingleSiteLogMRW(Parameter& param, const double tuning);
    //! Creates a block update group
    AdaptiveMultiLogMRW*
    newAdaptiveMultiLogMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    AdaptiveMultiMRW*
    newAdaptiveMultiMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    SpeciesMRW*
    newSpeciesMRW(const string tag, UpdateBlock& params, std::vector<double>& alpha, const size_t burnin = 1000);
    InfectivityMRW*
    newInfectivityMRW(const string tag, UpdateBlock& params, UpdateBlock& powers, const size_t burnin = 1000);
    SusceptibilityMRW*
    newSusceptibilityMRW(const string tag, UpdateBlock& params, UpdateBlock& powers, const size_t burnin = 1000);
    InfectivityPowMRW*
    newInfectivityPowMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    SusceptibilityPowMRW*
    newSusceptibilityPowMRW(const string tag, UpdateBlock& params, const size_t burnin = 1000);
    SellkeSerializer*
    newSellkeSerializer(const string filename);
    void
    setNumIUpdates(const size_t n);
    void
    calcLogLikelihood(Likelihood& logLikelihood);
    DIC
    getDIC();

  };

}
#endif
