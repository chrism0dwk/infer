/*************************************************************************
 *  ./src/Framework/Model.hpp
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
 * Model.hpp
 *
 *  Created on: Jun 14, 2011
 *      Author: stsiab
 */

#ifndef MODEL_HPP_
#define MODEL_HPP_

#include "Parameter.hpp"


namespace EpiRisk
{

  /*! Class representing an epidemic model
   *
   * This class represents an epidemic model, providing an interface
   * for algorithms to obtain infectious pressure, as well as to the
   * populations.
   */

  // TODO: Parameters needs to be templatized!!
  template<typename Population>
  class Model
  {
  public:

    typedef Population PopulationType;
    typedef typename Population::Individual Individual;

    Model(Population& population) : population_(population) {};
    virtual
    ~Model() {};

    /*! (Time dependent) Infectivity of individual i
     *
     * @param i the individual
     * @param time the time since i's infection for which infectivity is required.
     * @return the infectivity of i at time.
     */
    virtual
    double
    infectivity(const Individual& i, double time) const = 0;

    /*! Returns pressure between Infected and Susceptible
     *
     * @param i is the Infected individual
     * @param j is the Susceptible individual
     * @return instantaneous infectious pressure
     */
    virtual
    double
    beta(const Individual& i, const Individual& j, const double time) const = 0;

    /*! Returns pressure between Notified and Susceptible
     *
     * @param i is the Notified individual
     * @param j is the Susceptible individual
     * @return instantaneous infectious pressure
     */
    virtual
    double
    betastar(const Individual& i, const Individual& j, const double time) const = 0;

    /*! Returns the background pressure on an individual
     *
     * @param j the individual
     * @return the instantaneous background pressure exerted on an individual
     */
    virtual
    double
    background(const Individual& j) const = 0;

    virtual
    double
    instantPressureOn(const typename Population::InfectiveIterator& j, double time) const
    {
      double sumPressure = 0.0;
      typename Population::InfectiveIterator i = population_.infecBegin();
      typename Population::InfectiveIterator stop = population_.infecLowerBound(time); // Don't need people infected after time.

      while (i != stop)
        {
          if (i != j)
            { // Skip i==j
              if (i->getN() > time)
                {
                  sumPressure += beta(*i, *j, time);
                }
              else if (i->getR() > time)
                {
                  sumPressure += betastar(*i, *j, time);
                }
            }
          ++i;
        }
      sumPressure += background(*j);

      return sumPressure;
    }

    virtual
    double
    ItoN(const double rn) const = 0;

    virtual
    double
    NtoR() const = 0;

    Population&
    getPopulation() const { return population_; };

    Parameters*
    getParameters() const { return &params_; };

    double
    getObsTime() const {return population_.getObsTime();}



  protected:
    Parameters params_;
    Population& population_;

  };

}

#endif /* MODEL_HPP_ */
