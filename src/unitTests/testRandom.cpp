/*************************************************************************
 *  ./src/unitTests/testRandom.cpp
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
 * testRandom.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: stsiab
 */

#include <iostream>
#include <fstream>
#include <string>
#include <boost/tokenizer.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "Random.hpp"

int
main(int argc, char* argv[])
{
  using namespace boost::numeric::ublas;
  using namespace EpiRisk;
  Random::CovMatrix covar(3, 3);

  std::string buff;
  std::ifstream inFile;
  inFile.open("/Users/stsiab/testPDMatrix.txt", std::ios::in);

  size_t i, j;
  for (i = 0; i < 3; ++i)
    {
      std::cerr << i << std::endl;
      getline(inFile, buff);
      boost::tokenizer<boost::char_separator<char> > tok(buff,
          boost::char_separator<char>(" "));
      boost::tokenizer<boost::char_separator<char> >::iterator beg;
      for (j = 0, beg = tok.begin(); j < 3; ++j, ++beg)
        covar(i, j) = atof(beg->c_str());
    }

  Random myRandom(1);
  Random::Variates variates(3);
  Random::Variates proposal(3);
  for(size_t i=0; i<3; ++i) {
      variates(i) = 0.0;
  }

  for(size_t i=0; i<1000000; ++i) {
      proposal = myRandom.mvgauss(variates,covar * 2.38*2.38 / 3);
      for(size_t j=0;j<3;++j) std::cout << proposal(j) << "\t";
      std::cout << "\n";
  }
}
