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
