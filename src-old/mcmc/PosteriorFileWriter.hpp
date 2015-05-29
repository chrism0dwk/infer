/*************************************************************************
 *  ./src/mcmc/PosteriorFileWriter.hpp
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
 * PosteriorFileWriter.hpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#ifndef POSTERIORFILEWRITER_HPP_
#define POSTERIORFILEWRITER_HPP_

#include <string>

class PosteriorFileWriter
{
public:
  PosteriorFileWriter(const std::string filename, const EpiRisk::Parameters& parameters);
  virtual
  ~PosteriorFileWriter();
  virtual
  void
  open();
  virtual
  void
  close();


private:
  EpiRisk::Parameters& parameters_;
  std::ofstream fh_;
  std::string filename_;
};

#endif /* POSTERIORFILEWRITER_HPP_ */
