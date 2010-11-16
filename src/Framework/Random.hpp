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
    gaussian(const double mu, const double var);
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
  };

}

#endif /* RANDOM_HPP_ */
