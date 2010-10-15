/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

/* Header file for aifuncs.cpp */

#ifndef INCLUDE_MCMC_H
#define INCLUDE_MCMC_H

#include <math.h>
#include <limits>


#include "SpatPointPop.hpp"

#define NEGINF (-numeric_limits<double>::infinity)

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

  Parameters&
  Parameters(Parameters& toCopy)
  {
    Parameters newParms;
    return newParms;
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
  double beta(size_t i, size_t j);
  double betastar(size_t i, size_t j);

  void
  calcLogLikelihood();

public:
  Mcmc(Population<TestCovars>& population, Parameters& parameters);
  ~Mcmc();

  double
  getLikelihood() const;
};

#endif
