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


#include "SpatPointPop.hpp"
#include "Data.hpp"

#define NEGINF (-numeric_limits<double>::infinity())

using namespace std;
using namespace EpiRisk;

class Parameter
{
  double val_;

public:
  Parameter(double value)
  {
    val_ = value;
  }

  double
  operator()() const
  {
    return val_;
  }
};


struct Parameters
{
  Parameter* beta1;
  Parameter* beta2;
  Parameter* phi;

  Parameters()
  {
    beta1 = new Parameter(0.5);
    beta2 = new Parameter(0.3);
    phi = new Parameter(0.6);
  }

  Parameters(Parameters& toCopy)
  {
    beta1 = new Parameter(*toCopy.beta1);
    beta2 = new Parameter(*toCopy.beta2);
    phi = new Parameter(*toCopy.phi);
  }

  ~Parameters()
  {
    delete beta1;
    delete beta2;
    delete phi;
  }
};


class Mcmc {

  Population<TestCovars>& pop_;
  Parameters& params_;
  double logLikelihood_;
  double beta(const Population<TestCovars>::const_iterator i, const Population<TestCovars>::const_iterator j) const;
  double betastar(const Population<TestCovars>::const_iterator i, const Population<TestCovars>::const_iterator j) const;

  void
  calcLogLikelihood();

public:
  Mcmc(Population<TestCovars>& population, Parameters& parameters);
  ~Mcmc();

  double
  getLikelihood() const;
};

#endif
