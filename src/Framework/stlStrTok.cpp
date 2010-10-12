/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

// Function definition for STL std::string parser.
// C. Jewell, Lancaster University, Copyright 2007

#include "stlStrTok.hpp"

using namespace std;


void stlStrTok(vector<string>& tokens, const string myString, const char* delim)
{
  // Usage: vector<string> tokens - vector to contain parsed tokens
  //        const string myString - STL std::string containing string to be parsed
  //        char* delim - delimiter for parsing

  istringstream ss(myString);
  string token;

  tokens.clear();

  while(!ss.eof()) {
    getline(ss,token,*delim);
    tokens.push_back(token);
  }

}
