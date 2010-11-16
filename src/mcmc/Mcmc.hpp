/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

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
  double logLikelihood_;
  Random* random_;
  EmpCovar<LogTransform>* logTransCovar_;
  ublas::matrix<double>* stdCov_;
  map<string,double> productCache_;
  map<string,double> productCacheTmp_;

  int mpirank_,mpiprocs_;
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
  bool
  updateTrans();
  bool
  updateI(const size_t index);
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
