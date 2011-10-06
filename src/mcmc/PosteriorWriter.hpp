/*************************************************************************
 *  ./src/mcmc/PosteriorWriter.hpp
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
 * PosteriorWriter.hpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#ifndef POSTERIORWRITER_HPP_
#define POSTERIORWRITER_HPP_

#include "Parameter.hpp"

namespace EpiRisk
{

  class PosteriorWriter
  {
  public:
    virtual
    void
    open() = 0;
    virtual
    void
    close() = 0;
    virtual
    void
    write() = 0;
  };


  class PosteriorNullWriter
  {
  public:
    virtual
    void
    open() { };
    virtual
    void
    close() { };
    virtual
    void
    write(const Parameters& parameters) { }
  };

}

#endif /* POSTERIORWRITER_HPP_ */
