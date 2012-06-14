/*************************************************************************
 *  ./src/data/PosteriorReader.cpp
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
 * PosteriorReader.cpp
 *
 *  Created on: 29 Sep 2011
 *      Author: stsiab
 */

#include <cstdlib>
#include <map>
#include <vector>
#include "PosteriorReader.hpp"
#include "stlStrTok.hpp"
#include "EpiRiskException.hpp"

namespace EpiRisk
{

  PosteriorReader::PosteriorReader(const std::string parameters,
      const std::string infectionTimes)
  {
    // Open parameter file
    posteriorFile_.open(parameters.c_str(), ios::in);
    if (!posteriorFile_.is_open())
      {
        std::string msg = "Could not open parameter file '";
        msg += parameters;
        msg += "' for reading.";
        throw data_exception(msg.c_str());
      }
    std::string line;
    std::getline(posteriorFile_, line);

    stlStrTok(paramNames_, line, ",");
    if (paramNames_.size() == 0)
      throw parse_exception("Posterior file appears to be empty!");

    // Strip quote characters
    for(std::vector<std::string>::iterator it = paramNames_.begin();
        it != paramNames_.end();
        it++)
      {
        std::string tmp = *it;
        if(tmp[0] == '"') tmp.erase(0,1);
        if(tmp[tmp.size()-1] == '"') tmp.erase(tmp.size()-1,tmp.npos);
        *it = tmp;
      }

    infecFile_.open(infectionTimes.c_str(), ios::in);
    if (!infecFile_.is_open())
      {
        std::string msg = "Could not open parameter file '";
        msg += infectionTimes;
        msg += "' for reading.";
        throw data_exception(msg.c_str());
      }
  }

  PosteriorReader::~PosteriorReader()
  {
    posteriorFile_.close();
    infecFile_.close();
  }

  bool
  PosteriorReader::next(const size_t stride)
  {
    params_.clear();
    std::string line;
    std::vector<std::string> toks;
    for(size_t i=0; i<stride; ++i) std::getline(posteriorFile_,line);
    if(posteriorFile_.eof() or line.size() < 2) return false;
    stlStrTok(toks, line, ",");
    for(size_t i=0; i<toks.size(); ++i)
      {
        params_.insert(std::make_pair(paramNames_[i],atof(toks[i].c_str())));
      }

    infecTimes_.clear();
    for(size_t i=0; i<stride; ++i) std::getline(infecFile_,line);
    if(infecFile_.eof() or line.size() < 2) return false;
    stlStrTok(toks, line, " ");
    for(size_t i=0; i<toks.size(); ++i)
      {
        std::vector<std::string> crumbs;
        stlStrTok(crumbs, toks[i],":");
        if(crumbs[0] == "") break;
        if(crumbs.size() != 2) throw parse_exception("Malformed infection time tuple!");
        infecTimes_.insert(std::make_pair(crumbs[0],atof(crumbs[1].c_str())));
      }

    return true;
  }

  const std::map<std::string,double>&
  PosteriorReader::params()
  {
    return params_;
  }

  const std::map<std::string,double>&
  PosteriorReader::infecTimes()
  {
    return infecTimes_;
  }
}


