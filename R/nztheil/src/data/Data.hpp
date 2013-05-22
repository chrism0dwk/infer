/*************************************************************************
 *  ./src/data/Data.hpp
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

namespace EpiRisk {
struct TestCovars {
  double x;
  double y;
  double cattle;
  double pigs;
  double sheep;
  double goats;
  double deer;
  double cattleinf;
  double pigsinf;
  double sheepinf;
  double cattlesusc;
  double pigssusc;
  double sheepsusc;
};


struct COOMatrixElement
{
  string i;
  string j;
  float distance;
};


class PopDataImporter : public EpiRisk::DataImporter<TestCovars>
{
private:
  ifstream dataFile_;
  string filename_;

public:
  PopDataImporter(const string filename);
  PopDataImporter();
  virtual ~PopDataImporter();
  virtual
  void open();
  virtual
  void close();
  virtual
  Record next();
  virtual
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
  virtual
  void open();
  virtual
  void close();
  virtual
  Record next();
  virtual
  void reset();
};


class DistMatrixImporter : public EpiRisk::DataImporter<COOMatrixElement>
{
private:
  ifstream matrixFile_;
  string filename_;
public:
  DistMatrixImporter(const string filename);
  virtual ~DistMatrixImporter();
  void open();
  void close();
  Record next();
  void reset();
};

}


#endif /* IMPORTERS_HPP_ */
