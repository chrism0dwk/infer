/*************************************************************************
 *  ./src/Framework/types.hpp
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


// Types used for epidemic MCMC code


#ifndef INCLUDE_AITYPES_HPP
#define INCLUDE_AITYPES_HPP

#include <limits>

namespace EpiRisk
{

  typedef double eventTime_t;
  typedef unsigned int Ilabel_t;
  typedef unsigned int Slabel_t;
  typedef unsigned int Spos_t;
  typedef unsigned int Ipos_t;
  typedef float freq_t;

  const double POSINF( std::numeric_limits<double>::infinity() );
  const double NEGINF(-std::numeric_limits<double>::infinity() );

  typedef struct {
    size_t idx;
    float  val;
  } IPTuple_t;

}
#endif
