/*************************************************************************
 *  ./src/Framework/stlStrTok.cpp
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
/* ./src/Framework/stlStrTok.cpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */

// Function definition for STL std::string parser.
// C. Jewell, Lancaster University, Copyright 2007

#include "stlStrTok.hpp"

using namespace std;


void stlStrTok(std::vector<string>& tokens, const std::string myString, const char* delim)
{
  // Usage: vector<string> tokens - vector to contain parsed tokens
  //        const string myString - STL std::string containing string to be parsed
  //        char* delim - delimiter for parsing

  std::istringstream ss(myString);
  std::string token;

  tokens.clear();

  while(!ss.eof()) {
    getline(ss,token,*delim);
    tokens.push_back(token);
  }

}
