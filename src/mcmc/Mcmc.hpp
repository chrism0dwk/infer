/*************************************************************************
 *  ./src/mcmc/Mcmc.hpp
 *  Copyright Chris Jewell <chrism0dwk@gmail.com> 2012
 *
 *  This file is part of nztheileria.
 *
 *  nztheileria is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  nztheileria is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nztheileria.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************/

/* Header file for aifuncs.cpp */

//TODO Need to create an "output" class: as we add updaters, register the parameter (block) with the outputter.
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
#include "Parameter.hpp"
#include "Random.hpp"
#include "EmpCovar.hpp"
#include "GpuLikelihood.hpp"
#include "McmcLikelihood.hpp"


namespace EpiRisk
{
  namespace Mcmc
  {

    class Mcmc
         {
         public:
           explicit
           Mcmc();
           virtual
           ~Mcmc() {};
           void
           Register(LikelihoodHandler* logLikelihood, Random* random);
           virtual
           void
           SetTag(TagType tag);
           virtual
           TagType
           GetTag() const;
           virtual
           void
           Update() = 0;
           virtual
           std::map<std::string, float>
           GetAcceptance() const = 0;
           virtual
           void
           ResetAcceptance() = 0;

         protected:
           LikelihoodHandler* likelihood_;
           Random* random_;
           TagType tag_;
         };

    class McmcContainer : public Mcmc
    {
    public:
      virtual
      Mcmc*
      Create(const UpdaterType updaterType, TagType tag);
      virtual
      void
      Update();
      virtual
      std::map<std::string, float>
      GetAcceptance() const;
      virtual
      void
      ResetAcceptance();

    protected:
      boost::ptr_list<Mcmc> updateStack_;
    };

    class McmcRoot : public McmcContainer
    {
    public:
      explicit
      McmcRoot(GpuLikelihood& likelihood, const size_t seed);
      ~McmcRoot();
    };





  }
}
#endif
