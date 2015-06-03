/*************************************************************************
 *  ./src/Framework/Random.hpp
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
/* ./src/Framework/Random.hpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of nztheileria.
 *
 * nztheileria is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * nztheileria is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nztheileria.  If not, see <http://www.gnu.org/licenses/>. 
 */
/*
 * Random.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: stsiab
 */

#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace EpiRisk
{
  using namespace boost::numeric::ublas;
  using namespace boost::numeric;

  class Random
  {
    gsl_rng* rng_;
  public:

    typedef matrix<double> CovMatrix;
    typedef ublas::vector<double> Variates;

    Random(const unsigned long int seed);
    virtual
    ~Random();
    double
    gaussian(const double mean, const double var);
    double
    gamma(const double shape, const double rate);
    double
    beta(const double a, const double b);
    double
    uniform(const double a=0, const double b=1);
    size_t
    integer(const size_t n);
    Variates
    mvgauss(const CovMatrix& covariance);
    Variates
    mvgauss(const Variates& mu, const CovMatrix& covariance);
    double
    extreme(const double a, const double b);
    double
    gaussianTail(const double mean, const double var);
    Variates
    dirichlet(const Variates& alpha);
  };

}

#endif /* RANDOM_HPP_ */
