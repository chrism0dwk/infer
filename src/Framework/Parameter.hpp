/*************************************************************************
 *  ./src/Framework/Parameter.hpp
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
 * Parameter.hpp
 *
 *  Created on: Oct 20, 2010
 *      Author: stsiab
 */

#ifndef PARAMETER_HPP_
#define PARAMETER_HPP_

namespace EpiRisk
{
  /*! \brief Interface to EpiRisk prior
   *
   * The Prior class provides an interface to a prior distribution
   * used by the Parameter class.
   */
  class Prior
  {
  public:
    Prior() {};
    virtual
    ~Prior() {};

    /*! Returns the value of the prior pdf
     * evaluated at x
     * @param x The value of the prior parameter
     * @return The value of \f$f(x)$\f
     */
    virtual
    double
    operator()(const double& x) {};

  };

  class Parameter
  {

    double value_;
    Prior prior_;

  public:
    Parameter(const double value,Prior prior) : value_(value), prior_(prior) {}
    virtual
    ~Parameter() {};
    Prior& prior()
    {
      return prior_;
    }

    double operator()()
    {
      return value_;
    }
  };

}

#endif /* PARAMETER_HPP_ */
