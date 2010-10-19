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

#include "Mcmc.hpp"

using namespace EpiRisk;

inline
double
dist(const double x1, const double y1, const double x2, const double y2)
{
  double dx = x1 - x2;
  double dy = y1 - y2;

  return sqrt(dx*dx + dy*dy);
}

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& parameters) : pop_(population),
                                                                         params_(parameters),
                                                                         logLikelihood_(NEGINF)
{
   calcLogLikelihood();
}

Mcmc::~Mcmc()
{
  // Nothing to do at present
}


double
Mcmc::getLogLikelihood() const
{
  return logLikelihood_;
}

inline
double
Mcmc::beta(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x,i.getCovariates().y,j.getCovariates().x,j.getCovariates().y);
  distance *= distance;
  double phisq = (*params_.phi)() * (*params_.phi)();
  return (*params_.beta1)() * phisq / (phisq + distance);
}

inline
double
Mcmc::betastar(const Population<TestCovars>::Individual& i, const Population<TestCovars>::Individual& j) const
{
  double distance = dist(i.getCovariates().x,i.getCovariates().y,j.getCovariates().x,j.getCovariates().y);
  distance *= distance;
  double phisq = (*params_.phi)() * (*params_.phi)();
  return (*params_.beta2)() * phisq / (phisq + distance);
}


void
Mcmc::calcLogLikelihood()
{
  // Calculates log likelihood

  Population<TestCovars>::InfectiveIterator j = ++pop_.infecBegin();
  Population<TestCovars>::InfectiveIterator stop;
  logLikelihood_ = 0.0;

  // First calculate the log product
  while(j != pop_.infecEnd()) {
      double sumPressure = 0.0;
      Population<TestCovars>::InfectiveIterator i = pop_.infecBegin();
      stop = pop_.infecUpperBound(j->getI()); // Don't need people infected after me.

      while(i != stop)
        {
          if(i != j) { // Skip i==j

              if(i->getN() > j->getI())
                {
                  //cout << "I->S infection:(i:" << i->getId() << "), Ii=" << i->getI() << ", Ni = " << i->getN() << ", Ij=" << j->getI() << endl;;
                  sumPressure += beta(*i,*j);
                }
              else if (i->getR() > j->getI())
                {
                  //cout << "N->S infection\n";
                  sumPressure += betastar(*i,*j);
                }
            }
            ++i;
          }
      logLikelihood_ += log(sumPressure);
      ++j;
  }

  // Now calculate the integral
  Population<TestCovars>::InfectiveIterator i = pop_.infecBegin();
  Population<TestCovars>::PopulationIterator k = pop_.begin();

  double totalIntegPress = 0.0;
  while (k != pop_.end()) {
      double integPressure = 0.0;

      i = pop_.infecBegin();
      stop = pop_.infecLowerBound(k->getI()); // Don't need people infected after k
      while (i != stop) {
      // Infective -> Susceptible pressure
      integPressure += beta(*i, *k) * min(i->getN(),k->getI()) - min(i->getI(),k->getI());

      // Notified -> Susceptible pressure
      integPressure += betastar(*i, *k) * min(i->getR(),k->getI()) - min(i->getN(), k->getI());
      ++i;
      }
      ++k;
      totalIntegPress -= integPressure;
  }

  logLikelihood_ -= totalIntegPress;
}
