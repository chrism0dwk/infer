/*************************************************************************
 *  ./src/mcmc/PosteriorWriter.cpp
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
 * PosteriorWriter.cpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#include "PosteriorWriter.hpp"



namespace EpiRisk
{

  PosteriorWriter::PosteriorWriter(GpuLikelihood& likelihood) : likelihood_(likelihood)
  {
  }

  PosteriorWriter::~PosteriorWriter()
  {
  }

  void
  PosteriorWriter::AddParameter(Parameter& param)
  {
    paramVals_.push_back(&param);
    paramTags_.push_back(param.GetTag());
  }


}
