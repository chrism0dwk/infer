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


FmdModel::FmdModel(Population<TestCovars>& population, FmdParameters& parameters)
  : Model< Population<TestCovars> >(population), params_(parameters)
{

}

FmdModel::~FmdModel()
{
}

double
FmdModel::infectivity(const Individual& i, const double time) const
{

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
  double dsq = dx*dx + dy*dy;
  double pressure = 0.0f;
  if ( dsq <= 25.0*25.0 )
    pressure = params_.gamma1 * params_.delta / powf(params_.delta*params_.delta + dsq, params_.omega);

  return pressure;
}

double
FmdModel::beta(const Individual& i, const Individual& j, const double time) const
{
  double val = 0.0;
  val += time < params_.movtban ? params_.epsilon1 : params_.epsilon2;

  val += distance(i,j);
  
  return val * infectivity(i,time) * susceptibility(j);
}

double
FmdModel::betastar(const Individual& i, const Individual& j, const double time) const
{
  double val = params_.gamma2 * distance(i,j);
  return val;
}

double
FmdModel::background(const Individual& j, const double t) const
{
  return 1e-10; //params_.epsilon1 * (t > params_.movtban ? params_.epsilon2 : 1.0);
}

double
FmdModel::hFunction(const Individual& j, const double time) const
{
  assert(time >= 0.0);
	if(time - j.getI() < params_.latency) return 0.0;
	else return 1.0;
}


double
FmdModel::ItoN(Random& random) const
{
  return random.gamma(params_.a, params_.b);
}

double
FmdModel::NtoR(Random& random) const
{
  return random.gamma(params_.c,params_.d);
}

double
FmdModel::leftTruncatedItoN(Random& random, const Individual& j) const
{
  EpiRisk::FP_t d = population_.getObsTime() - j.getI();
  EpiRisk::FP_t s = gsl_cdf_gamma_P(d, params_.a, 1.0/params_.b);
  EpiRisk::FP_t u = random.uniform(s,1.0);
  d = gsl_cdf_gamma_Pinv(u, params_.a, 1.0/params_.b);

  return d;
}

double
FmdModel::leftTruncatedNtoR(Random& random, const Individual& j) const
{
  EpiRisk::FP_t d = population_.getObsTime() - j.getN();
  EpiRisk::FP_t s = gsl_cdf_gamma_P(d, params_.c, 1.0/params_.d);
  EpiRisk::FP_t u = random.uniform(s,1.0);
  d = gsl_cdf_gamma_Pinv(u, params_.a, 1.0/params_.b);
  return d;
}
