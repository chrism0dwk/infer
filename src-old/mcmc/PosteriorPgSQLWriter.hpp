/*************************************************************************
 *  ./src/mcmc/PosteriorPgSQLWriter.hpp
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
 * PosteriorPgSQLWriter.hpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#ifndef POSTERIORPGSQLWRITER_HPP_
#define POSTERIORPGSQLWRITER_HPP_

#include <string>
#include <libpg-fe.h>
#include "PosteriorWriter.hpp"

class PosteriorPgSQLWriter
{
public:
  PosteriorPgSQLWriter(const std::string connString, const EpiRisk::Parameters);
  virtual
  ~PosteriorPgSQLWriter();
};

#endif /* POSTERIORPGSQLWRITER_HPP_ */
