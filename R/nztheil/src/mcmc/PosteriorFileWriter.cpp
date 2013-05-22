/*************************************************************************
 *  ./src/mcmc/PosteriorFileWriter.cpp
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
 * PosteriorFileWriter.cpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#include "PosteriorFileWriter.hpp"
#include "EpiRiskException.hpp"

PosteriorFileWriter::PosteriorFileWriter(const std::string filename, const EpiRisk::Parameters& parameters) :
  filename_(filename),parameters_(parameters),fh_(NULL)
{

}

PosteriorFileWriter::~PosteriorFileWriter()
{
  if(fh_ != NULL) fh_.close();
}

void
PosteriorFileWriter::open()
{
  fh_.open(filename_.c_str(),std::ios::out);
  if(!fh_.is_open()) {
      std::string msg = "Cannot open posterior output file '";
      msg += fh_;
      msg += "' for writing";
      throw EpiRisk::output_exception(msg.c_str());
  }

  fh_ << parameters_(0).getTag();
  for(EpiRisk::Parameters::iterator it = ++parameters_.begin();
      it != parameters_.end();
      it++)
    {
      fh_ << "," << it->getTag();
    }
  fh_ << "\n";
}

void
PosteriorFileWriter::close()
{
  if(fh_ != NULL) fh_.close();
}

void
PosteriorFileWriter::write()
{
  fh_ << parameters_(0).getTag();
  for(EpiRisk::Parameters::iterator it = ++parameters_.begin();
      it != parameters_.end();
      it++)
    {
      fh_ << "," << it->getTag();
    }
  fh_ << "\n";
}
