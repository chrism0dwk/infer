/*************************************************************************
 *  ./src/unitTests/fmdModel.cpp
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
/*
 * fmdModel.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: stsiab
 */

#include <cassert>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include "fmdModel.hpp"




TheileriaModel::TheileriaModel(Population<TestCovars>& population, TheileriaParameters& parameters)
  : Model< Population<TheileriaCovars> >(population), params_(parameters)
{

}

TheileriaModel::~TheileriaModel()
{
}

double
TheileriaModel::infectivity(const Individual& i, const double time) const
{

  double infectivity = 1;

  return infectivity;
}

double
TheileriaModel::susceptibility(const Individual& j) const
{
  double susceptibility = j.getCovariates().ticks;

  return susceptibility;
}

double
TheileriaModel::distance(const Individual& i, const Individual& j) const
{
  double dx = i.getCovariates().x - j.getCovariates().x;
  double dy = i.getCovariates().y - j.getCovariates().y;
  double distance = sqrt( dx*dx + dy*dy );

  return distance;
}

double
TheileriaModel::beta(const Individual& i, const Individual& j, const double time) const
{
  double dist = distance(i,j);
  if ( dist <= 25.0 ) {
      return params_.phi * infectivity(i,time) * susceptibility(j) * params_.delta / (params_.delta*params_.delta + dist*dist)^1.5;
  }
  else return 0.0;
}

double
TheileriaModel::betastar(const Individual& i, const Individual& j, const double time) const
{
  double dist = distance(i,j);
  if ( dist <= 25.0 ) {
      return params_.gamma2 * params_.gamma1 * infectivity(i,time) * susceptibility(j) * params_.delta / (params_.delta*params_.delta + dist*dist);
  }
  else return 0.0;
}

double
TheileriaModel::background(const Individual& j, const double t) const
{
  return params_.epsilon1 * (t > params_.movtban ? params_.epsilon2 : 1.0);
}

double
TheileriaModel::hFunction(const Individual& j, const double time) const
{
  assert(time >= 0.0);
	if(time - j.getI() < params_.latency) return 0.0;
	else return 1.0;
}


double
TheileriaModel::ItoN(Random& random) const
{
  return random.gamma(params_.a, params_.b);
}

double
TheileriaModel::NtoR() const
{
  return params_.ntor;
}

double
TheileriaModel::leftTruncatedItoN(Random& random, const Individual& j) const
{
  EpiRisk::FP_t d = population_.getObsTime() - j.getI();
  EpiRisk::FP_t s = gsl_cdf_gamma_P(d, params_.a, 1.0/params_.b);
  EpiRisk::FP_t u = random.uniform(s,1.0);
  d = gsl_cdf_gamma_Pinv(u, params_.a, 1.0/params_.b);

  return d;
}

