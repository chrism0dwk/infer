/*************************************************************************
 *  ./src/mcmc/Mcmc.cpp
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

#include <cmath>
#include <boost/numeric/ublas/io.hpp>
#include <ctime>
#include <algorithm>
#include <functional>
#include <vector>
#include <iomanip>
#include <gsl/gsl_cdf.h>
#include <sys/time.h>

//#ifdef __LINUX__
//#include <acml_mv.h>
//#define log fastlog
//#define exp fastexp
//#endif

#include "Mcmc.hpp"
#include "McmcFactory.hpp"

namespace EpiRisk
{
  namespace Mcmc
  {

// Constants
    const double tuneI = 2.5;

    inline
    double
    timeinseconds(const timeval a, const timeval b)
    {
      timeval result;
      timersub(&b, &a, &result);
      return result.tv_sec + result.tv_usec / 1000000.0;
    }

    inline
    double
    onlineMean(const double x, const double xbar, const double n)
    {
      return xbar + (x - xbar) / n;
    }

    Mcmc::Mcmc() : likelihood_(NULL),random_(NULL) {}

    void
    Mcmc::Register(LikelihoodHandler* logLikelihood, Random* random)
    {
      likelihood_ = logLikelihood;
      random_ = random;
    }

    void
    Mcmc::SetTag(TagType tag)
    {
      tag_ = tag;
    }

    TagType
    Mcmc::GetTag() const
    {
      return tag_;
    }

//! Creates an updater and pushes it onto the MCMC call stack
    Mcmc*
    McmcContainer::Create(const UpdaterType updaterType, TagType tag)
    {
      Mcmc* updater = McmcFactory::Instance().Create(updaterType);
      updater->Register(likelihood_, random_);
      updater->SetTag(tag);
      updateStack_.push_back(updater);
      return updater;
    }

//! Performs a parameter sweep
    void
    McmcContainer::Update()
    {
      // Performs a sweep over the parameters
          for (boost::ptr_list<Mcmc>::iterator it = updateStack_.begin();
              it != updateStack_.end(); ++it)
            {
              it->Update();
            }
    }

    map<string, float>
    McmcContainer::GetAcceptance() const
    {
      map<string, float> acceptance;
      for (boost::ptr_list<Mcmc>::const_iterator it =
          updateStack_.begin(); it != updateStack_.end(); ++it)
        {
          map<string, float> tmp = it->GetAcceptance();
          acceptance.insert(tmp.begin(), tmp.end());
        }

      return acceptance;
    }

    void
    McmcContainer::ResetAcceptance()
    {
      for (boost::ptr_list<Mcmc>::iterator it = updateStack_.begin();
          it != updateStack_.end(); ++it)
        it->ResetAcceptance();
    }


    McmcRoot::McmcRoot(GpuLikelihood& likelihood, size_t seed)
    {
      likelihood_ = new LikelihoodHandler(likelihood);
      random_ = new Random(seed);
      tag_ = "Root";
    }

    McmcRoot::~McmcRoot()
    {
      delete likelihood_;
      delete random_;
    }



  }
}
