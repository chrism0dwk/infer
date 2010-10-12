/*
 * Importers.hpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */

#ifndef IMPORTERS_HPP_
#define IMPORTERS_HPP_

#include <fstream>
#include <string>

#include "Individual.hpp"
#include "DataImporter.hpp"

struct TestCovars {
  long int x;
  long int y;
  long int herdSize;
};


class PopDataImporter : public EpiRisk::DataImporter<TestCovars>
{
private:
  ifstream dataFile_;
public:
  PopDataImporter(const string filename);
  virtual ~PopDataImporter();
  Record next();
  void reset();
};


class EpiDataImporter : public EpiRisk::DataImporter<EpiRisk::Events>
{
private:
  ifstream dataFile_;
public:
  EpiDataImporter(const string filename);
  virtual ~EpiDataImporter();
  Record next();
  void reset();
};

#endif /* IMPORTERS_HPP_ */
