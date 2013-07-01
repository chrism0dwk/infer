/*
 * McmcFactory.cpp
 *
 *  Created on: Oct 16, 2012
 *      Author: cpjewell
 */

#include <stdexcept>

#include "McmcFactory.hpp"

namespace EpiRisk
{
  namespace Mcmc
  {
    namespace
     {
       Mcmc*
       CreateSingleSiteLogMRW()
       {
         return new SingleSiteLogMRW;
       }
       Mcmc*
       CreateAdaptiveSingleMRW()
       {
	 return new AdaptiveSingleMRW;
       }
       Mcmc*
       CreateAdaptiveMultiMRW()
       {
         return new AdaptiveMultiMRW;
       }
       Mcmc*
       CreateAdaptiveMultiLogMRW()
       {
         return new AdaptiveMultiLogMRW;
       }
       Mcmc*
       CreateInfectivityMRW()
       {
         return new InfectivityMRW;
       }
       Mcmc*
       CreateSusceptibilityMRW()
       {
         return new SusceptibilityMRW;
       }
       Mcmc*
       CreateInfectionTimeGammaCentred()
       {
         return new InfectionTimeGammaCentred;
       }
       Mcmc*
       CreateInfectionTimeGammaNC()
       {
         return new InfectionTimeGammaNC;
       }
       Mcmc*
       CreateInfectionTimeUpdate()
       {
         return new InfectionTimeUpdate;
       }
     }

      /////// Provided Updater registrations
      void Initialize()
      {
      const bool singleSiteLogMRW = McmcFactory::Instance().RegisterUpdater(
          "SingleSiteLogMRW", CreateSingleSiteLogMRW);

      const bool adaptiveSingleMRW = McmcFactory::Instance().RegisterUpdater(
	  "AdaptiveSingleMRW", CreateAdaptiveSingleMRW);

      const bool adaptiveMultiMRW = McmcFactory::Instance().RegisterUpdater(
          "AdaptiveMultiMRW", CreateAdaptiveMultiMRW);

      const bool adaptiveMultiLogMRW = McmcFactory::Instance().RegisterUpdater(
          "AdaptiveMultiLogMRW", CreateAdaptiveMultiLogMRW);

      const bool infectivityMRW = McmcFactory::Instance().RegisterUpdater(
          "InfectivityMRW", CreateInfectivityMRW);

      const bool susceptibilityMRW = McmcFactory::Instance().RegisterUpdater(
          "SusceptibilityMRW", CreateSusceptibilityMRW);

      const bool infectionTimeGammaCentred =
          McmcFactory::Instance().RegisterUpdater("InfectionTimeGammaCentred",
              CreateInfectionTimeGammaCentred);

      const bool infectionTimeGammaNC = McmcFactory::Instance().RegisterUpdater(
          "InfectionTimeGammaNC", CreateInfectionTimeGammaNC);

      const bool infectionTimeUpdate = McmcFactory::Instance().RegisterUpdater(
          "InfectionTimeUpdate", CreateInfectionTimeUpdate);
    }




    McmcFactory* McmcFactory::pInstance_ = 0;

    McmcFactory::McmcFactory()
    {
      std::cout << "Initialising MCMC factory" << std::endl;
    }

    McmcFactory::~McmcFactory()
    {
      delete pInstance_;
    }
    McmcFactory&
    McmcFactory::Instance()
    {
      if (!pInstance_)
        {
          pInstance_ = new McmcFactory;
        }
      return *pInstance_;
    }

    bool
    McmcFactory::IsInitialized()
    {
      return pInstance_ != 0;
    }

    bool
    McmcFactory::RegisterUpdater(const UpdaterType updaterId,
        const CreateMcmcUpdater createFn)
    {
      return updaters_.insert(UpdaterMap::value_type(updaterId, createFn)).second;
    }

    bool
    McmcFactory::UnregisterUpdater(const UpdaterType updaterId)
    {
      return updaters_.erase(updaterId) == 1;
    }

    Mcmc*
    McmcFactory::Create(const UpdaterType updaterId)
    {
      UpdaterMap::iterator i = updaters_.find(updaterId);
      if (i == updaters_.end())
        throw std::runtime_error("Unknown updater ID");
      std::cerr << "Created updater " << updaterId << "\n";
      return (i->second)();
    }

  }
} /* namespace EpiRisk */
