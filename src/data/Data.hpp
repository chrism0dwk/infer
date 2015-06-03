/*************************************************************************
 *  ./src/data/Data.hpp
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
 * Importers.hpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */

#ifndef IMPORTERS_HPP_
#define IMPORTERS_HPP_

#include <fstream>
#include <string>

#include "types.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"

struct TestCovars {
  double x;
  double y;
  double ticks;
  bool isDairy;
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
  EpiDataImporter();
  virtual ~EpiDataImporter();
  virtual void open();
  virtual void close();
  virtual Record next();
  virtual void reset();
};

struct COOMatrixElement
{
  std::string i;
  std::string j;
  EpiRisk::FP_t val;
};

class ContactDataImporter : public EpiRisk::DataImporter<COOMatrixElement>
{
public:
  virtual ~ContactDataImporter() {};
};

#endif /* IMPORTERS_HPP_ */
