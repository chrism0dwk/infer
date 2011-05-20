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
#include "SirDeterministic.cpp"

struct TestCovars {
  double x;
  double y;
  double horses;
  double area;
  SirDeterministic* epi;
};


class PopDataImporter : public EpiRisk::DataImporter<TestCovars>
{
private:
  ifstream dataFile_;
  string filename_;

public:
  PopDataImporter(const string filename);
  virtual ~PopDataImporter();
  void open();
  void close();
  Record next();
  void reset();
};


class EpiDataImporter : public EpiRisk::DataImporter<EpiRisk::Events>
{
private:
  ifstream dataFile_;
  string filename_;

public:
  EpiDataImporter(const string filename);
  virtual ~EpiDataImporter();
  void open();
  void close();
  Record next();
  void reset();
};





#endif /* IMPORTERS_HPP_ */
