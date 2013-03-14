/*************************************************************************
 *  ./src/mcmc/MCMCUpdater.hpp
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
/* ./src/mcmc/MCMCUpdater.hpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */
/*
 * MCMCUpdater.hpp
 *
 *  Created on: 25 Jan 2011
 *      Author: stsiab
 */

#ifndef MCMCUPDATER_HPP_
#define MCMCUPDATER_HPP_

#include <string>
#include "EmpCovar.hpp"
#include "Mcmc.hpp"


namespace EpiRisk
{
  namespace Mcmc
  {

#define ADAPTIVESCALE 1.0
#define WINDOWSIZE 100

    // FUNCTORS
    struct ExpTransform
    {
      float
      operator()(const float x)
      {
        return exp(x);
      }
    };

    struct LogTransform
    {
      float
      operator()(const float x)
      {
        return log(x);
      }
    };

    // MCMC UPDATERS
    class McmcUpdate : public Mcmc
    {
    public:
      McmcUpdate();
      virtual
      ~McmcUpdate();
      virtual
      void
      SetParameters(UpdateBlock& parameters);
      virtual
      void
      Update() = 0;

      //! Returns the MH acceptance probability
      virtual std::map<string, float>
      GetAcceptance() const;
      virtual
      void
      ResetAcceptance();

    protected:
      size_t acceptance_;
      size_t numUpdates_;
      UpdateBlock* params_;
    };

    class SingleSiteLogMRW : public McmcUpdate
    {
    public:
      SingleSiteLogMRW();
      void
      SetTuning(const float tuning);
      void
      Update();

    private:
      float tuning_;
    };


    //! Adaptive Multisite updater class
    template<class Transform>
      class AdaptiveMRW : public McmcUpdate
      {
      public:
        typedef typename EmpCovar<Transform>::CovMatrix Covariance;

        AdaptiveMRW() :
            burnin_(300), adaptScalar_(ADAPTIVESCALE), windowUpdates_(0), windowAcceptance_(
                0), isAdaptiveInitialized_(false), empCovar_(NULL), stdCov_(
                NULL)
        {
        }
        ~AdaptiveMRW()
        {
          if (empCovar_)
            delete empCovar_;
          if (stdCov_)
            delete stdCov_;
        }
        virtual
        void
        SetParameters(UpdateBlock& params)
        {
          params_ = &params;
          InitCovariance(params);
        }

        void
        SetBurnin(const size_t burnin)
        {
          burnin_ = burnin;
        }
        void
        SetCovariance(Covariance& covariance)
        {
          // Start the empirical covariance matrix
          delete empCovar_;
          empCovar_ = new EmpCovar<Transform>(*params_, covariance);
        }
        Covariance
        GetCovariance()
        {
          return empCovar_->getCovariance();
        }
        virtual
        void
        Update() = 0;

      protected:
        void
        InitCovariance(UpdateBlock& params)
        {
          if (empCovar_)
            delete empCovar_;
          if (stdCov_)
            delete stdCov_;

          // Initialize the standard covariance
          stdCov_ = new Covariance(params.size());
          for (size_t i = 0; i < params.size(); ++i)
            {
              for (size_t j = 0; j < params.size(); ++j)
                {
                  if (i == j)
                    (*stdCov_)(i, j) = 0.01 / params.size();
                  else
                    (*stdCov_)(i, j) = 0.0;
                }
            }

          empCovar_ = new EmpCovar<Transform>(params, *stdCov_);
        }
        size_t burnin_;
        float adaptScalar_;
        size_t windowUpdates_;
        size_t windowAcceptance_;
        bool isAdaptiveInitialized_;
        EmpCovar<Transform>* empCovar_;
        typename EmpCovar<Transform>::CovMatrix* stdCov_;
      };

    
    class AdaptiveSingleMRW : public AdaptiveMRW<Identity>
    {
    public:
      void
      Update();
    };


    //! Adaptive Multisite Linear Random Walk algorithm
    class AdaptiveMultiMRW : public AdaptiveMRW<Identity>
    {
    public:
      typedef EmpCovar<Identity>::CovMatrix Covariance;
      void
      Update();
    };

    //! Adaptive Multisite Logarithmic Random Walk algorithm
    class AdaptiveMultiLogMRW : public AdaptiveMRW<LogTransform>
    {
    public:
      void
      Update();
    };

    //! InfectivityMRW is a non-centred Multisite update for species infectivity
    class InfectivityMRW : public AdaptiveMRW<LogTransform>
    {
    public:
      void
      SetParameters(UpdateBlock& params);
      void
      Update();

    private:
      UpdateBlock transformedGroup_;
      std::vector<float> constants_;
    };

    //! SusceptibilityMRW is a non-centred Multisite update for species inf/susc
    class SusceptibilityMRW : public AdaptiveMRW<LogTransform>
    {
    public:
      void
      SetParameters(UpdateBlock& params);
      void
      Update();

    private:
      UpdateBlock transformedGroup_;
      std::vector<float> constants_;

    };

    //! InfectionTimeGammaScale performs centred updating of a gamma infectious period scale parameters
    class InfectionTimeGammaCentred : public McmcUpdate
    {
    public:
      explicit
      InfectionTimeGammaCentred();
      void
      SetTuning(const float tuning);
      void
      Update();

    private:
      float tuning_;
      size_t windowUpdates_;
      size_t windowAcceptance_;
    };

    //! InfectionTimeGammaNC performed partially non-centred updating of a gamma infectious period scale parameter
    class InfectionTimeGammaNC : public McmcUpdate
    {
    public:
      explicit
      InfectionTimeGammaNC();
      virtual
      ~InfectionTimeGammaNC();
      void
      SetNCRatio(const float ncProp);
      void
      SetTuning(const float tuning);
      void
      Update();
    private:
      float tuning_;
      float ncProp_;
      size_t windowUpdates_;
      size_t windowAcceptance_;
    };

    //! InfectionTimeUpdate performs an update or rj move
    class InfectionTimeUpdate : public McmcUpdate
    {
    public:
      explicit
      InfectionTimeUpdate();
      virtual
      ~InfectionTimeUpdate();
      void
      SetUpdateTuning(const float tuning)
      {
    	  updateTuning_ = tuning;
      }
      void
      SetReps(const size_t reps);
      void
      SetCompareProductVector(bool* doCompareProductVector)
      {
        doCompareProductVector_ = doCompareProductVector;
      }
      void
      SetOccults(const bool doOccults) {
	doOccults_ = doOccults;
      }
      void
      Update();
      std::map<std::string, float>
      GetAcceptance() const;
      void
      ResetAcceptance();
    private:
      bool* doCompareProductVector_;
      bool doOccults_;
      ublas::vector<float> calls_;
      ublas::vector<float> accept_;
      size_t reps_;
      size_t ucalls_;
      float updateTuning_;
      bool
      UpdateI();
      bool
      AddI();
      bool
      DeleteI();

    };

  }
}
#endif /* MCMCUPDATER_HPP_ */
