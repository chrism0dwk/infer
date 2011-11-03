/*
 * Importers.cpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cstdlib>
#include <limits>
#include "Data.hpp"
#include "stlStrTok.hpp"

#define NEGINF (-numeric_limits<double>::infinity())
#define POSINF (numeric_limits<double>::infinity())

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
        throw EpiRisk::data_exception("Cannot open population file for reading");
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
  if (tokens.size() != 6) throw EpiRisk::fileEOF();

  record.id = tokens[0];
  record.data.x = atof(tokens[1].c_str()) / 1000.0;
  record.data.y = atof(tokens[2].c_str()) / 1000.0;
  record.data.horses = atof(tokens[3].c_str());
  record.data.area = atof(tokens[4].c_str());
  record.data.vaccdate = atof(tokens[5].c_str());
  record.data.epi = new EpiRisk::SirDeterministic(record.data.horses,0.25);

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

EpiDataImporter::~EpiDataImporter()
{
  dataFile_.close();
}

void
EpiDataImporter::open()
{
  dataFile_.open(filename_.c_str(),ios::in);
      if(!dataFile_.is_open()) {
          throw EpiRisk::data_exception("Cannot open population file for reading");
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
  if (tokens.size() != 3) throw EpiRisk::fileEOF();
  record.id = tokens[0];
  record.data.I = atof(tokens[1].c_str());
  record.data.N = POSINF;//atof(tokens[2].c_str());
  record.data.R = POSINF;//atof(tokens[3].c_str());
  return record;
}


void
EpiDataImporter::reset()
{
  dataFile_.seekg(0);
  string row;
  getline(dataFile_,row);
}
