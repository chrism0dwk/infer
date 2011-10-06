/*************************************************************************
 *  ./release/brandenburg-0.3alpha/src/Framework/stlStrTok.hpp
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

// Header file for STL std::string parser. Provides a wrapper for strtok.
// C. Jewell, Lancaster University, Copyright 2007

#ifndef STLSTRTOK_H
#define STLSTRTOK_H

#include <stdexcept>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

void stlStrTok(vector<string>& tokens, const string myString, const char* delim = "\n");

#endif
