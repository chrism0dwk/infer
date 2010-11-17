/*************************************************************************
 *  ./src/Framework/McmcWriter.hpp
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
 * DataWriter.hpp
 *
 *  Created on: Nov 1, 2010
 *      Author: stsiab
 */

#ifndef DATAWRITER_HPP_
#define DATAWRITER_HPP_

#include <fstream>

#include "SpatPointPop.hpp"
#include "Parameter.hpp"

namespace EpiRisk
{

  template<class Population>
  class McmcWriter
  {
  private:
    ofstream paramFile_;
    ofstream occFile_;
    string paramFn_, occFn_;

  public:
    McmcWriter(const string paramFile,
               const string occultFile) : paramFn_(paramFile), occFn_(occultFile)
    {

    }
    virtual
    ~McmcWriter()
    {
      if(paramFile_.is_open()) paramFile_.close();
      if(occFile_.is_open()) occFile_.close();
    }
    virtual
    void
    open()
    {
      paramFile_.open(paramFn_.c_str(),ios::out);
      if(!paramFile_.is_open()) throw output_exception("Cannot open parameter file for writing!");

      occFile_.open(occFn_.c_str(),ios::out);
      if(!paramFile_.is_open()) throw output_exception("Cannot open occult file for writing!");
    }
    virtual
    void
    close()
    {
      if(paramFile_.is_open()) paramFile_.close();
      if(occFile_.is_open()) occFile_.close();
    }
    virtual
    void
    write(Population& population)
    {
      // Write population here
      typename Population::InfectiveIterator it = population.infecBegin();
      while(it != population.infecEnd()) {
          occFile_ << it->getId() << ":" << it->getI() << " ";
          it++;
      }
      occFile_ << "\n";
    }
    virtual
    void
    write(Parameters& params)
    {
      // Write parameters here;
      Parameters::iterator it = params.begin();
      paramFile_ << (double)*it;
      it++;
      while(it != params.end())
        {
          paramFile_ << "," << (double)*it;
          it++;
        }
      paramFile_ << "\n";
    }
  };

}

#endif /* DATAWRITER_HPP_ */
