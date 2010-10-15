/*
 * Importers.cpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */

#include <cstdlib>
#include "Data.hpp"
#include "stlStrTok.hpp"


PopDataImporter::PopDataImporter(const string filename)
{
  dataFile_.open(filename.c_str(),ios::in);
  if(!dataFile_.is_open()) {
      throw EpiRisk::data_exception("Cannot open population file for reading");
  }

  // Take out header line
  string row;
  getline(dataFile_,row);

}

PopDataImporter::~PopDataImporter()
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
  if (tokens.size() != 4) throw EpiRisk::fileEOF();

  record.id = tokens[0];
  record.data.x = atoi(tokens[1].c_str());
  record.data.y = atoi(tokens[2].c_str());
  record.data.herdSize = atoi(tokens[3].c_str());

  return record;
}

void
PopDataImporter::reset()
{
  dataFile_.seekg(0);
  string row; getline(dataFile_,row);
}



EpiDataImporter::EpiDataImporter(const string filename)
{

    dataFile_.open(filename.c_str(),ios::in);
    if(!dataFile_.is_open()) {
        throw EpiRisk::data_exception("Cannot open population file for reading");
    }

    string row;
    getline(dataFile_,row);
}

EpiDataImporter::~EpiDataImporter()
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
  if (tokens.size() != 4) throw EpiRisk::fileEOF();

  record.id = tokens[0];
  record.data.I = atof(tokens[1].c_str());
  record.data.N = atof(tokens[2].c_str());
  record.data.R = atof(tokens[3].c_str());

  return record;
}


void
EpiDataImporter::reset()
{
  dataFile_.seekg(0);
  string row;
  getline(dataFile_,row);
}
