/*************************************************************************
 *  ./src/unitTests/testEmpCov.cpp
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
 * testEmpCov.cpp
 *
 *  Created on: 23 Dec 2010
 *      Author: stsiab
 */

#include <iostream>
#include <fstream>
#include <boost/tokenizer.hpp>

#include "EmpCovar.hpp"

struct LogTransform
{
  double operator()(const double x)
  {
    return log(x);
  };
};

int main(int argc, char* argv[])
{

  EmpCovar
}
