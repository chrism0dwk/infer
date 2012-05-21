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

#include <gsl/gsl_randist.h>

#include "fmdModel.hpp"


FmdModel::FmdModel(Population<TestCovars>& population, FmdParameters& parameters) : Model< Population<TestCovars> >(population), params_(parameters)
{

}

FmdModel::~FmdModel()
{

}

double
FmdModel::infectivity(const Individual& i, const double time) const
{

  if(time - i.getI() < params_.latency) return 0.0;

  double infectivity = powf(i.getCovariates().cattle,params_.psi_c) +
                       params_.xi_p*powf(i.getCovariates().pigs,params_.psi_p) +
                       params_.xi_s*powf(i.getCovariates().sheep,params_.psi_s);

  return infectivity;
}

double
FmdModel::susceptibility(const Individual& j) const
{
  double susceptibility = powf(j.getCovariates().cattle,params_.phi_c) +
                          params_.zeta_p*powf(j.getCovariates().pigs,params_.phi_p) +
                          params_.zeta_s*powf(j.getCovariates().sheep,params_.phi_s);

  return susceptibility;
}

double
FmdModel::distance(const Individual& i, const Individual& j) const
{
  double dx = i.getCovariates().x - j.getCovariates().x;
  double dy = i.getCovariates().y - j.getCovariates().y;
  double distance = sqrt( dx*dx + dy*dy );

  return distance;
}

double
FmdModel::beta(const Individual& i, const Individual& j) const
{
  double dist = distance(i,j);
  if ( dist <= 25.0 ) {
      return params_.gamma1 * infectivity(i,0.0) * susceptibility(j) * params_.delta / (params_.delta*params_.delta + dist*dist);
  }
  else return 0.0;
}

double
FmdModel::betastar(const Individual& i, const Individual& j) const
{
  double dist = distance(i,j);
  if ( dist <= 25.0 ) {
      return params_.gamma2 * params_.gamma1 * infectivity(i,0.0) * susceptibility(j) * params_.delta / (params_.delta*params_.delta + dist*dist);
  }
  else return 0.0;
}

double
FmdModel::background(const Individual& j) const
{
  return params_.epsilon;
}

double
FmdModel::ItoN(const double rn) const
{
  return rn;
}

double
FmdModel::ItoN(Random& random) const
{
  return random.gamma(params_.a, params_.b);
}

double
FmdModel::NtoR() const
{
  return params_.ntor;
}
