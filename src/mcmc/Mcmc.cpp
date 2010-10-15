/*************************************************************************
 *  ./src/mcmc/Mcmc.cpp
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

#include "Mcmc.cpp"

using EpiRisk::Population<TestCovars>;

double
dist(const double x1, const double y1, const double x2, const double y2)
{
  double dx = x1 - x2;
  double dy = y1 - y2;

  return sqrt(dx*dx + dy*dy);
}

Mcmc::Mcmc(Population<TestCovars>& population, Parameters& parameters) : pop_(population),
                                                                         params_(parameters),
                                                                         logLiklihood_(NEGINF)
{
   // Constructor
}

Mcmc::~Mcmc()
{
  // Nothing to do at present
}

double
Mcmc::getLikelihood() const
{
  return logLiklihood_;
}

double
Mcmc::beta(const Population<TextCovars>::const_iterator i, const Population<TestCovars>::const_iterator j) const
{
  return params_.beta1() * exp(-params_.phi() * dist(i->getX(),i->getY(),j->getX(),j->getY()));
}

double
Mcmc::betastar(const Population<TestCovars>::const_iterator i, const Population<TestCovars>::const_iterator j) const
{
  return params_.beta2() * exp(-params_.phi() * dist(i->getX(),i->getY(),j->getX(),j->getY()));
}


void
Mcmc::calcLogLikelihood() const
{
  // Calculates log likelihood

  Infectives& infectives = pop_.getInfectives();
  Infectives::iterator i = infectives.begin();
  Infectives::iterator j = infectives.begin();

  logLikelihood_ = 0.0;

  // First calculate the log product
  while(j != infectives.end()) {
      double sumPressure = 0.0;
      Infectives::iterator i = infectives.begin();
      Infectives::iterator stop = infectives.upper_bound(j->getI()); // Don't need people infected after me.
      while(i != stop)
        {
          if(i->getN() > j->getI())
            {
              sumPressure += beta(*i,*j);
            }
          else if (i->getR() > j->getI())
            {
              sumPressure += betastar(*i,*j);
            }
          ++i;
        }
      logLikelihood_ += log(sumPressure);
      ++j;
  }


  // Now calculate the integral
}







}
