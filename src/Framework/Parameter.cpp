/*************************************************************************
 *  ./src/Framework/Parameter.cpp
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
 * Parameter.cpp
 *
 *  Created on: Mar 20, 2012
 *      Author: stsiab
 */

#include "Parameter.hpp"

namespace EpiRisk
{
  std::ostream&
  operator<<(std::ostream& os, const ParameterSerializer& paramSerializer)
  {
    ParameterSerializerList::const_iterator it =
        paramSerializer.params_.begin();
    os << it->GetValue();
    ++it;
    while (it != paramSerializer.params_.end())
      {
        os << "," << it->GetTag();
        ++it;
      }
    return os;
  }

}
