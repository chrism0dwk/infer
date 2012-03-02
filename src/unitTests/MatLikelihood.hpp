/*************************************************************************
 *  ./src/unitTests/MatLikelihood.hpp
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
 * MatLikelihood.hpp
 *
 *  Created on: 30 Jan 2012
 *      Author: stsiab
 */

#ifndef MATLIKELIHOOD_HPP_
#define MATLIKELIHOOD_HPP_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


#include "Parameter.hpp"
#include "SpatPointPop.hpp"
#include "Data.hpp"
#include "GpuLikelihood.hpp"

using namespace boost::numeric;
using namespace boost::numeric::ublas;

class MatLikelihood
{

  typedef float fp_t;

public:
  MatLikelihood(const EpiRisk::Population<TestCovars>& population,EpiRisk::Parameters& txparams);
  virtual
  ~MatLikelihood();
  double
  calculate();
  double
  gpuCalculate();


private:
  std::map<size_t,size_t> rawtoblas_;
  const EpiRisk::Population<TestCovars>& population_;
  EpiRisk::Parameters& txparams_;
  fp_t obsTime_;
  ublas::vector<fp_t> icoord_; // Array of i (row) coordinates
  ublas::vector<fp_t> jcoord_; // Array of j (column) coordinates
  size_t infectivesSz_;
  size_t subPopSz_;
  matrix<fp_t,column_major> animals_; // n*m matrix of m species on n farms
  matrix<fp_t,column_major> animalsInfPow_; // Powers of m species on n farms for infectivity
  matrix<fp_t,column_major> animalsSuscPow_; // For susceptibility
  matrix<fp_t,column_major> eventTimes_; // n*3 matrix of I,N,R times for each n infected farm
  matrix_range< matrix<fp_t,column_major> >* infecTimes_;
  fp_t I1_;
  ublas::vector<fp_t>::size_type I1idx_;
  ublas::vector<fp_t> infectivity_; // Infectivities
  ublas::vector<fp_t> susceptibility_; // Susceptibilities
  ublas::vector<fp_t> product_; // Product cache
  typedef compressed_matrix<fp_t> spm_t;
  spm_t D_; // Spatial kernel matrix
  compressed_matrix<fp_t,row_major> E_; // Exposed at time of infection (product)
  spm_t T_; // Exposure time
  spm_t DT_; // Spatial kernel * exposure time
  ublas::unbounded_array<int> DRowPtr_;
  ublas::unbounded_array<int> DColInd_;
  ublas::unbounded_array<int> ERowPtr_;
  ublas::unbounded_array<int> EColPtr_;

  typedef EpiRisk::Population<TestCovars>::Individual Individual;
  typedef std::vector<size_t> SubPopulation;

  // Constants
  const float zero_;
  const float unity_;

  GpuLikelihood* gpu_;

};

#endif /* MATLIKELIHOOD_HPP_ */
