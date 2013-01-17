/*
 * McmcFactory.hpp
 *
 *  Created on: Oct 16, 2012
 *      Author: cpjewell
 */

#ifndef MCMCFACTORY_HPP_
#define MCMCFACTORY_HPP_

#include <map>
#include <string>
#include <McmcLikelihood.hpp>
#include <boost/ptr_container/ptr_list.hpp>

#include "types.hpp"
#include "MCMCUpdater.hpp"

namespace EpiRisk
{

  class Random;

  namespace Mcmc
  {

    class Mcmc;

    class McmcFactory
    {
    public:
      typedef Mcmc*
      (*CreateMcmcUpdater)();

      static McmcFactory&
      Instance();
      static bool
      IsInitialized();
      bool
      RegisterUpdater(const UpdaterType updaterId,
          const CreateMcmcUpdater createFn);
      bool
      UnregisterUpdater(const UpdaterType UpdaterId);
      Mcmc*
      Create(const UpdaterType updaterId);

    private:
      McmcFactory(); // Prevent public from calling this
      McmcFactory(const McmcFactory&); // Prevent clients creating copies
      McmcFactory&
      operator=(const McmcFactory&);
      ~McmcFactory();

      static McmcFactory* pInstance_;
      typedef std::map<UpdaterType, CreateMcmcUpdater> UpdaterMap;
      UpdaterMap updaters_;
    };

    void
    Initialize();

  } /* namespace Mcmc */
} /* namespace EpiRisk */
#endif /* MCMCFACTORY_HPP_ */
