/*************************************************************************
 *  ./src/data/Data.cpp
 *  Copyright Chris Jewell <chrism0dwk@gmail.com> 2012
 *
 *  This file is part of nztheileria.
 *
 *  nztheileria is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  nztheileria is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nztheileria.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************/

/*
 * Importers.cpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cstdlib>
#include <vector>
#include <sstream>
#include "types.hpp"
#include "Data.hpp"
#include "stlStrTok.hpp"


gsl_rng* localrng = gsl_rng_alloc(gsl_rng_mt19937);


PopDataImporter::PopDataImporter(const string filename) : filename_(filename)
{

}

PopDataImporter::~PopDataImporter()
{
  if(dataFile_.is_open()) dataFile_.close();
}

void
PopDataImporter::open()
{
  dataFile_.open(filename_.c_str(),ios::in);
    if(!dataFile_.is_open()) {
        std::stringstream msg;
        msg << "Cannot open population file '" << filename_ << "' for reading";
        throw EpiRisk::data_exception(msg.str().c_str());
    }

    // Take out header line
    string row;
    getline(dataFile_,row);

}

void
PopDataImporter::close()
{
  dataFile_.close();
}


PopDataImporter::Record
PopDataImporter::next()
{
  string row;
  Record record;
  vector<string> tokens;

  if(dataFile_.eof())
      throw EpiRisk::fileEOF();

  getline(dataFile_,row);

  stlStrTok(tokens,row,",");
  if (tokens.size() < 4) throw EpiRisk::fileEOF();

  record.id = tokens[0];
  record.data.x = atof(tokens[1].c_str()) / 1000;
  record.data.y = atof(tokens[2].c_str()) / 1000;
  record.data.ticks = atof(tokens[3].c_str());
  record.data.isDairy = atoi(tokens[6].c_str());
  
  return record;
}

void
PopDataImporter::reset()
{
  dataFile_.seekg(0);
  string row; getline(dataFile_,row);
}



EpiDataImporter::EpiDataImporter(const string filename) : filename_(filename)
{

}

EpiDataImporter::EpiDataImporter()
{
}

EpiDataImporter::~EpiDataImporter()
{
  if(dataFile_.is_open())
    dataFile_.close();
}

void
EpiDataImporter::open()
{
  dataFile_.open(filename_.c_str(),ios::in);
      if(!dataFile_.is_open()) {
          std::stringstream msg;
          msg << "Cannot open epidemic file '" << filename_ << "' for reading";
          throw EpiRisk::data_exception(msg.str().c_str());
      }

  string row;
  getline(dataFile_,row);
}

void
EpiDataImporter::close()
{
  dataFile_.close();
}

EpiDataImporter::Record
EpiDataImporter::next()
{
  string row;
  Record record;
  vector<string> tokens;

  if(dataFile_.eof()) throw EpiRisk::fileEOF();

  getline(dataFile_,row);

  stlStrTok(tokens,row,",");
  if (tokens.size() < 4) throw EpiRisk::fileEOF();

  record.id = tokens[0];
  if (tokens[1] == "Inf") record.data.I = EpiRisk::POSINF;
  else if(tokens[1] == "") record.data.I = EpiRisk::POSINF;
  else record.data.I = atof(tokens[1].c_str());

  if(tokens[2] == "Inf") record.data.N = EpiRisk::POSINF;
  else if(tokens[2] == "") record.data.N = EpiRisk::POSINF;
  else record.data.N = atof(tokens[2].c_str());

  if(tokens[3] == "Inf") record.data.R = EpiRisk::POSINF;
  else if(tokens[3] == "") record.data.R = EpiRisk::POSINF;
  else record.data.R = atof(tokens[3].c_str());

  record.data.type = tokens[4];

  return record;
}


void
EpiDataImporter::reset()
{
  dataFile_.seekg(0);
  string row;
  getline(dataFile_,row);
}
