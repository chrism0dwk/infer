/*************************************************************************
 *  ./src/Framework/Random.cpp
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
 * Random.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: stsiab
 */

#include <stdexcept>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include "Random.hpp"
#include "EpiRiskException.hpp"


namespace EpiRisk
{

  Random::Random(const unsigned long int seed)
  {
    rng_ = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng_,seed);

  }

  Random::~Random()
  {
    gsl_rng_free(rng_);
  }
  double
  Random::gaussian(const double mean, const double var)
  {
    return mean + gsl_ran_gaussian(rng_, sqrt(var));
  }
  double
  Random::gamma(const double shape, const double rate)
  {
    return gsl_ran_gamma(rng_, shape, 1 / rate);
  }
  double
  Random::beta(const double a, const double b)
  {
    return gsl_ran_beta(rng_, a, b);
  }
  double
  Random::uniform(const double a, const double b)
  {
    return gsl_ran_flat(rng_, a, b);
  }
  size_t
  Random::integer(const size_t n)
  {
    return gsl_rng_uniform_int(rng_, n);
  }
  Random::Variates
  Random::mvgauss(const CovMatrix& covariance)
  {

    // Cholesky decomposition to get SD matrix
    if(covariance.size1() != covariance.size2())
        throw std::invalid_argument(
            "Cholesky decomposition is only valid for a square, positive definite matrix.");

    CovMatrix sigma = covariance;
    size_t size = sigma.size1();
    Variates d(size);
    for (size_t i = 0; i < size; ++i)
      {
        matrix_row < CovMatrix > mri(row(sigma, i));
        for (size_t j = i; j < size; ++j)
          {
            matrix_row < CovMatrix > mrj(row(sigma, j));
            double elem = sigma(i, j) - inner_prod(project(mri, range(0, i)),
                project(mrj, range(0, i)));
            if (i == j)
              {
                if (elem <= 0.0)
                  {
                    throw cholesky_error("Matrix after rounding errors is not positive definite");
                  }
                else
                  {
                    d(i) = sqrt(elem);
                  }
              }
            else
              {
                sigma(j, i) = elem / d(i);
              }
          }
      }
    // put the diagonal back in
    for (size_t i = 0; i < size; ++i)
      {
        sigma(i, i) = d(i);
      }


    // Generate the variates
    vector<double> variates(size);
    for(size_t i=0; i<size; ++i) variates(i) = gaussian(0,1);

    return prod(sigma,variates);
  }
  Random::Variates
  Random::mvgauss(const Variates& mu, const CovMatrix& covariance)
  {
    // Generates MVN(0,Sigma) variates
    return mu + mvgauss(covariance);
  }
  double
  Random::extreme(const double a, const double b)
  {
    return 1.0/b * log(1-log(1-uniform())/a);
  }
  double
  Random::gaussianTail(const double mean, const double var)
  {
    return gsl_ran_gaussian_tail(rng_,-mean,sqrt(var));
  }

}
