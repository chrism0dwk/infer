/*************************************************************************
 *  ./src/unitTests/testPosteriorReader.cpp
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
 * testPosteriorReader.cpp
 *
 *  Created on: 29 Sep 2011
 *      Author: stsiab
 */

#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include "PosteriorReader.hpp"

int
main(int argc, char* argv[])
{
  EpiRisk::PosteriorReader posterior(argv[1], argv[2]);

  while (posterior.next())
    {
      for ( std::map<std::string, double>::const_iterator it =
          posterior.params().begin(); it != posterior.params().end(); it++)
        std::cout << it->first << ":" << it->second << "\t";

      std::cout << std::endl;

      for( std::map< std::string, double >::const_iterator it = posterior.infecTimes().begin();
          it != posterior.infecTimes().end();
          it++)
        std::cout << it->first << ":" << it->second << "\t";

      std::cout << std::endl;

      if (!posterior.next())
        break;
    }

  return EXIT_SUCCESS;
}
