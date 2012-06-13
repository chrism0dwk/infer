/* ./src/Framework/EmpCovar.hpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */

/////////////////////////////////////////////////////////////////////
// Name: Adaptive v. 2.0                                           //
// Author: C.P.Jewell                                              //
// Purpose: AdaptiveRW provides a proposal mechanism for an        //
//          adaptive random walk.                                  //
/////////////////////////////////////////////////////////////////////


/* CODE EXAMPLE - 5 parameters, update every 100 iterations
 MCMC length is 100000 iterations
 #include <gsl/gsl_matrix.h>
 #include "adaptive.h"

 double mySigma[] = {1,1,1,1,1}
 McmcOutput myOutput(5,100,10000,mySigma}

 //MCMC Loop:
 for(int mcmcIter = 0; mcmcIter < 100000; ++mcmcIter) {

 // Generate proposal:

 gsl_matrix* propSD = McmcOutput.scaleChol(2.38)
 // Simulate from our proposal density using propSD

 // Accept or reject

 // Add the current state to the posterior

 McmcOutput.add(parms)
 }

 */

#ifndef _INCLUDE_ADAPTIVE_H
#define _INCLUDE_ADAPTIVE_H


#include <stdexcept>
#include <iostream>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "StochasticNode.hpp"

using namespace boost::numeric::ublas;
using namespace boost::numeric;

namespace EpiRisk
{

  class CholeskyFail : public std::exception
  {
  public:
    CholeskyFail(const char* msg)
    {
      msg_ = msg;
    }
    virtual const char*
    what() const throw ()
    {
      return msg_;
    }

  private:
    const char* msg_;

  };

  struct Identity
  {
    double
    operator()(const double x) const
    {
      return x;
    }
  };


  template<typename Transform=Identity>
    class EmpCovar
    {
    public:
      typedef symmetric_matrix<double> CovMatrix;

    private:
      typedef ublas::vector<Parameter>::const_iterator ParamIter; // Tries to enforce type consistency

      size_t p_;
      ublas::vector<double> sum_;
      symmetric_matrix<double> sumSq_;
      CovMatrix covMatrix_;
      ublas::vector<double> expectation_;

      const UpdateBlock& params_;

      int rowCount_;

      Transform transformFunc_;

      void
      kronecker()
      {
        for (size_t i = 0; i < expectation_.size(); ++i)
          for (size_t j = 0; j <= i; ++j)
            covMatrix_(i, j) = expectation_(i) * expectation_(j);
      }


    public:
      EmpCovar(const UpdateBlock& params, CovMatrix& covariance) :
        params_(params),p_(params.size()),
            rowCount_(0)
      {
        // Set up storage
        sum_.resize(params.size());
        sumSq_.resize(params.size());
        expectation_.resize(params.size());
        for(size_t i = 0; i<params.size(); ++i) {
            sum_(i) = 0.0;
            expectation_(i) = 0.0;
            for(size_t j = 0; j < params.size(); ++j) sumSq_(i,j) = 0.0;
        }
        covMatrix_ = covariance;


        // Add a parameter row
        //setCovariance(covariance);
        //sample();
      }
      ~EmpCovar()
      {
      }
      const CovMatrix&
      getCovariance()
      {
        // Create covariance matrix
        double denominator = rowCount_;
        expectation_ = sum_ / denominator; // Averages
        kronecker();
        covMatrix_ = sumSq_ / denominator - covMatrix_;
        return covMatrix_;
      }
      void
      printInnerds()
      {
        std::cerr << "Sum: " << sum_ << std::endl;
        std::cerr << "Sumsq: " << sumSq_ << std::endl;
        std::cerr << "Row count: " << rowCount_ << std::endl;
        expectation_ = sum_ / (double)rowCount_;
        std::cerr << "Expectation: " << expectation_ << std::endl;
        kronecker();
        std::cerr << "kronecker: " << covMatrix_ << std::endl;
        std::cerr << "covMatrix: " << sumSq_ / (double)rowCount_ - covMatrix_;
      }
      void
      sample()
      {
        for (int i = 0; i < params_.size();++i)
          {
            double pi = transformFunc_(params_[i]->getValue());
            sum_(i) += pi;
            sumSq_(i, i) += pi*pi;
            for (size_t j = 0; j < i; ++j)
              {
                double pj = transformFunc_(params_[j]->getValue());
                sumSq_(i, j) += pi * pj;
              }
          }

        rowCount_++;
      }
      void
      print();

    };

} // namespace EpiRisk


#endif
