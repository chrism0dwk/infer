/*************************************************************************
 *  ./src/unitTests/MatLikelihood.cpp
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
 * MatLikelihood.cpp
 *
 *  Created on: 30 Jan 2012
 *      Author: stsiab
 */
#include <set>
#include <vector>
#include <iterator>
#include <map>
#include <boost/numeric/ublas/operation.hpp>
#include <sys/time.h>

#include "MatLikelihood.hpp"


class CmpIndivIdxOnInfection
{
  const EpiRisk::Population<TestCovars>& population_;
public:
  CmpIndivIdxOnInfection(const EpiRisk::Population<TestCovars>& population) :
    population_(population)
  {
  }
  ;
  bool
  operator()(const size_t lhs, const size_t rhs) const
  {
    return population_[lhs].getI() < population_[rhs].getI();
  }
  ;
};

MatLikelihood::MatLikelihood(const EpiRisk::Population<TestCovars>& population,
    EpiRisk::Parameters& txparams) :
  txparams_(txparams),infectivesSz_(population.numInfected()), obsTime_(population.getObsTime()),
      population_(population)
{
  // Get all susceptibles relevant to current epidemic
  size_t nnz = 0; // Number of non-zero entries in K(i,j) matrix
  set<size_t> tmp; // Stored in infection time order

  // Get list of individuals involved in the epidemic
  for (int i = 0; i < population.size(); ++i)
    {
      tmp.insert(population[i].getConnectionList().begin(),
          population[i].getConnectionList().end());
      if (population[i].getI() < obsTime_)
        nnz += population[i].getConnectionList().size();
    }

  // Order subpopulation by infection time
  SubPopulation subpop(tmp.begin(),tmp.end());
  CmpIndivIdxOnInfection comp(population);
  sort(subpop.begin(),subpop.end(),comp);
  subPopSz_ = subpop.size();

  // Animals, susceptibility, infectivity, and D_ sparse matrix
  animals_.resize(subPopSz_, 3);
  animalsSuscPow_.resize(subPopSz_,3);
  animalsInfPow_.resize(infectivesSz_,3);
  susceptibility_.resize(subPopSz_);
  infectivity_.resize(infectivesSz_);
  product_.resize(infectivesSz_);
  icoord_.resize(infectivesSz_);
  jcoord_.resize(subPopSz_);
  infecTimes_.resize(infectivesSz_,3);

  E_.resize(infectivesSz_,infectivesSz_,false);
  D_.resize(infectivesSz_, subPopSz_,false);
  T_.resize(infectivesSz_, subPopSz_,false);

  cerr << "Populating data tables..." << endl;
  std::map<size_t,size_t> rawtoblas;
  size_t j_idx = 0;
  for (SubPopulation::const_iterator j = subpop.begin(); j != subpop.end(); ++j)
    {
      // Enter into jcoord
      jcoord_(j_idx) = *j;
      rawtoblas.insert(make_pair(*j,j_idx));

      // Copy covariates
      const Individual::CovarsType& covars = population_[*j].getCovariates();
      animals_(j_idx, 0) = covars.cattle;
      animals_(j_idx, 1) = covars.pigs;
      animals_(j_idx, 2) = covars.sheep;
      animalsSuscPow_(j_idx,0) = pow(covars.cattle,txparams_(13));
      animalsSuscPow_(j_idx,1) = pow(covars.pigs,txparams_(14));
      animalsSuscPow_(j_idx,2) = pow(covars.sheep,txparams_(15));
      j_idx++;
    }

  // Set up infectives
  cerr << "Setting up infectives" << endl;

  for(size_t i = 0; i < infectivesSz_; ++i)
    {

          icoord_(i) = jcoord_(i);
          infecTimes_(i, 0) = population_[jcoord_[i]].getI();
          infecTimes_(i, 1) = population_[jcoord_[i]].getN();
          infecTimes_(i, 2) = population_[jcoord_[i]].getR();

          for(size_t k = 0; k < 3; ++k)
            {
              animalsInfPow_(i,k) = powf(animals_(i,k),txparams_(k+10));
            }
    }

  // Cache I1
  matrix_column< matrix<fp_t> > col(infecTimes_,0);
  I1_ = col(0); I1idx_ = 0;
  for(size_t i = 1; i < col.size(); ++i)
    if (col(i) < I1_) {
        I1_ = col(i); I1idx_=i;
    }


  // Calculate product mask and exposure times
  for(size_t i = 0; i < infectivesSz_; ++i) {
          // Populate row of D_ and T_
          for (Individual::ConnectionList::const_iterator con =
              population_[jcoord_[i]].getConnectionList().begin(); con
              != population_[jcoord_[i]].getConnectionList().end(); ++con)
            {
              fp_t dx = population_[jcoord_[i]].getCovariates().x
                  - population_[*con].getCovariates().x;
              fp_t dy = population_[jcoord_[i]].getCovariates().y
                  - population_[*con].getCovariates().y;
              fp_t sqDist = dx * dx + dy * dy;

              // Get con's index in jcoord
              size_t j = rawtoblas[*con];

              // Product mask
              if(j < infectivesSz_ and i != j) {
                  if (infecTimes_(i,0) < infecTimes_(j,0) and infecTimes_(j,0) <= infecTimes_(i,1)) E_(i,j) = 1.0f;
                  else if (infecTimes_(i,1) < infecTimes_(j,0) and infecTimes_(j,0) <= infecTimes_(i,2)) E_(i,j) = txparams_(1);
              }

              // Integral of infection time
              fp_t jMaxSuscepTime = 0.0;
              if(j < infectivesSz_)
                {
                  if(infecTimes_(i,0) < infecTimes_(j,0)) jMaxSuscepTime = infecTimes_(j,0);
                }
              else
                {
                  jMaxSuscepTime = min((fp_t)population_[*con].getN(),obsTime_);
                }
              D_(i, j) = sqDist;
              fp_t exposureTime;
              exposureTime = min(infecTimes_(i,1),jMaxSuscepTime) - min(infecTimes_(i,0),jMaxSuscepTime);
              exposureTime += txparams_(1)*(min(infecTimes_(i,2),jMaxSuscepTime)-min(infecTimes_(i,1),jMaxSuscepTime));
              if (exposureTime < 0.0) cerr << "T_(" << population_[icoord_[i]].getId() << "," << population_[*con].getId() << ") = " << exposureTime << endl;
              T_(i, j) = exposureTime;
            }
    }

  DT_ = D_;
  cerr << "Initialised D_ with nnz=" << D_.nnz() << endl;
  cerr << "Initialised T_ with nnz=" << T_.nnz() << endl;
  cerr << "Initialised E_ with nnz=" << E_.nnz() << endl;
}

MatLikelihood::~MatLikelihood()
{

}

double
MatLikelihood::calculate()
{
  // Temporaries
  ublas::vector<fp_t> tmp(infectivesSz_);

  // Parameters
  fp_t deltasq = txparams_(2) * txparams_(2);
  ublas::vector<fp_t> infecParams(3);
  ublas::vector<fp_t> suscepParams(3);
  for (size_t i = 0; i < 3; ++i)
    {
      infecParams(i) = txparams_(i + 4);
      suscepParams(i) = txparams_(i + 7);
    }

  // Susceptibility
  axpy_prod(animalsSuscPow_, suscepParams, susceptibility_, true);

  // Infectivity
  axpy_prod(animalsInfPow_, infecParams, infectivity_, true);

  // Calculate product
  fp_t lp = 0.0;
  compressed_matrix<fp_t,column_major> QE(E_);
  for(size_t j = 0; j != QE.size1(); j++) // Iterate over COLUMNS j
    {
      fp_t subprod = 0.0;
      size_t begin = QE.index1_data()[j];
      size_t end = QE.index1_data()[j+1];
      for(size_t i = begin; i < end; ++i) // Non-zero ROWS i
        {
          QE.value_data()[i] *= txparams_(2) / (deltasq + D_(QE.index2_data()[i],j));
        }
    }
  axpy_prod(infectivity_,QE,tmp);

  for(size_t i = 0; i < I1idx_; ++i)
    {
      fp_t subprod = susceptibility_(i)*tmp(i) + txparams_(3);
      product_(i) = subprod; lp += logf(subprod);
    }
  product_(I1idx_) = 1.0; // Loop unrolled to skip I1
  for(size_t i = I1idx_+1; i < tmp.size(); ++i)
    {
      fp_t subprod = txparams_(0)*susceptibility_(i)*tmp(i) + txparams_(3);
      product_(i) = subprod; lp += logf(subprod);
    }

  // Apply distance kernel to D_ and calculate DT
  for(size_t i = 0; i < D_.size1(); ++i)
    {
      size_t begin = D_.index1_data()[i];
      size_t end = D_.index1_data()[i+1];

      for(size_t j = begin; j < end; ++j)
        {
          DT_.value_data()[j] = txparams_(2) / (deltasq + D_.value_data()[j]) * T_.value_data()[j];
        }
    }

  // Calculate the integral
  axpy_prod(DT_,susceptibility_,tmp);
  fp_t integral = txparams_(0) * inner_prod(infectivity_,tmp);

  // Calculate background pressure
  matrix_column< matrix<fp_t> > col(infecTimes_,0);
  fp_t bg = sum(col) - I1_*infectivesSz_;
  bg += (obsTime_ - I1_)*(population_.size() - infectivesSz_);
  bg *= txparams_(3);

  integral += bg;

  return lp - integral;
}
