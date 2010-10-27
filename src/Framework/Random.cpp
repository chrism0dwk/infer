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
  Random::gaussian(const double mu, const double var)
  {
    return mu + gsl_ran_gaussian(rng_, sqrt(var));
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
  Random::mvgauss(const Variates& mu,
      const CovMatrix& covariance)
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

    std::cout << "===========SIGMA MATRIX==========\n" << sigma << std::endl;


    // Generate the variates
    vector<double> variates(size);
    for(size_t i=0; i<size; ++i) variates(i) = gaussian(0,1);

    return mu+prod(sigma,variates);
  }

}
