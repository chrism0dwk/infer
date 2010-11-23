/*************************************************************************
 *  ./src/mcmc/Mcmc.hpp
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

/* Header file for aifuncs.cpp */

#ifndef INCLUDE_MCMC_H
#define INCLUDE_MCMC_H

#include <math.h>
#include <limits>
#include <iostream>
#include <fstream>


#include "SpatPointPop.hpp"
#include "Data.hpp"
#include "Parameter.hpp"
#include "Random.hpp"
#include "EmpCovar.hpp"
#include "McmcWriter.hpp"

#define NEGINF (-numeric_limits<double>::infinity())

using namespace std;
using namespace EpiRisk;

enum MpiMsgTag {
  MPIPRODCACHE = 0
};

struct ExpTransform
{
  double operator()(const double x)
  {
    return exp(x);
  }
};

struct LogTransform
{
  double operator()(const double x)
  {
    return log(x);
  }
};



class Mcmc {

  Population<TestCovars>& pop_;
  Parameters& params_;
  double logLikelihood_, logLikCan_, gLogLikelihood_, logProduct_, logProductCan_;
  Random* random_;
  EmpCovar<LogTransform>* logTransCovar_;
  ublas::matrix<double>* stdCov_;

  int mpirank_,mpiprocs_;
  bool mpiInitHere_;
  bool accept_;
  ofstream mcmcOutput_;

  virtual
  double
  beta(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const;
  virtual
  double
  betastar(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const;
  double
  instantPressureOn(const Population<TestCovars>::InfectiveIterator& j, const double Ij);
  double
  integPressureOn(const Population<TestCovars>::PopulationIterator& j, const double Ij);
  double
  calcLogLikelihood();
  double
  updateIlogLikelihood(const Population<TestCovars>::InfectiveIterator& j, const double newTime);
  double
  updateIlogLikelihoodSerial(const Population<TestCovars>::InfectiveIterator& j, const double newTime);
  bool
  updateTrans();
  bool
  updateI(const size_t index);
  void
  moveProdCache(const string id, const size_t fromIndex, const size_t toIndex);
  void
  dumpParms() const;
  void
  dumpProdCache();
public:
  Mcmc(Population<TestCovars>& population, Parameters& parameters, const size_t randomSeed);
  ~Mcmc();
  double
  getLogLikelihood() const;
  map<string,double>
  run(const size_t numIterations, McmcWriter<Population<TestCovars> >& writer);
};

#endif
