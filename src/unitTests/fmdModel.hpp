/*************************************************************************
 *  ./src/unitTests/fmdModel.hpp
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
 * fmdModel.hpp
 *
 *  Created on: Jun 14, 2011
 *      Author: stsiab
 */

#ifndef FMDMODEL_HPP_
#define FMDMODEL_HPP_

#include "Model.hpp"
#include "SpatPointPop.hpp"
#include "Data.hpp"
#include "Random.hpp"

using namespace EpiRisk;

  class GammaPrior : public Prior
  {
    float shape_;
    float rate_;
  public:
    GammaPrior(const float shape, const float rate)
    {
      shape_ = shape;
      rate_ = rate;
    }
    float
    operator()(const float x)
    {
      return gsl_ran_gamma_pdf(x,shape_,1/rate_);
    }
    Prior*
    create() const
    {
      return new GammaPrior(shape_,rate_);
    }
    Prior*
    clone() const
    {
      return new GammaPrior(*this);
    }
  };

  class BetaPrior : public Prior
  {
    float a_;
    float b_;
  public:
    BetaPrior(const float a, const float b) : a_(a),b_(b) {};
    float operator()(const float x)
    {
      return gsl_ran_beta_pdf(x,a_,b_);
    }
    Prior*
    create() const
    {
      return new BetaPrior(a_,b_);
    }
    Prior*
    clone() const
    {
      return new BetaPrior(*this);
    }
  };

struct FmdParameters
{
  Parameter gamma1;
  Parameter gamma2;
  Parameter delta;
  Parameter epsilon;
  Parameter xi_p;
  Parameter xi_s;
  Parameter psi_c;
  Parameter psi_p;
  Parameter psi_s;
  Parameter zeta_p;
  Parameter zeta_s;
  Parameter phi_c;
  Parameter phi_p;
  Parameter phi_s;
  Parameter a;
  Parameter b;
};


class FmdModel : public Model< Population<TestCovars> >
{
public:
  FmdModel(Population<TestCovars>& population, FmdParameters& parameters);
  virtual
  ~FmdModel();

  double distance(const Individual& i, const Individual& j) const;
  double infectivity(const Individual& i, const double time) const;
  double susceptibility(const Individual& j) const;

  double beta(const Individual& i, const Individual& j) const;
  double betastar(const Individual& i, const Individual& j) const;
  double background(const Individual& j) const;
  double ItoN(const double rn) const;
  double ItoN(Random& random) const;
  double NtoR() const;

private:
  FmdParameters& params_;
};

#endif /* FMDMODEL_HPP_ */
