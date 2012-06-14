/*************************************************************************
 *  ./src/data/PosteriorReader.hpp
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
 * PosteriorReader.hpp
 *
 *  Created on: 29 Sep 2011
 *      Author: stsiab
 */

#ifndef POSTERIORREADER_HPP_
#define POSTERIORREADER_HPP_

#include <string>
#include <map>
#include <vector>
#include <fstream>

#include "types.hpp"

namespace EpiRisk
{

  class PosteriorReader
  {
  public:
    PosteriorReader(const std::string parameters, const std::string infectionTimes);
    virtual
    ~PosteriorReader();

    const std::map<std::string,double>&
    params();

    const std::map<std::string,double>&
    infecTimes();

    bool
    next(const size_t stride=1);

  private:
    std::ifstream posteriorFile_;
    std::ifstream infecFile_;

    std::vector<std::string> paramNames_;
    std::map<std::string,double> params_;
    std::map<std::string,double> infecTimes_;
  };

}

#endif /* POSTERIORREADER_HPP_ */
