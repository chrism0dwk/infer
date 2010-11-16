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

  public:
    McmcWriter(const string paramFile,
               const string occultFile)
    {
      paramFile_.open(paramFile.c_str(),ios::out);
      if(!paramFile_.is_open()) throw output_exception("Cannot open parameter file for writing!");

      occFile_.open(occultFile.c_str(),ios::out);
      if(!paramFile_.is_open()) throw output_exception("Cannot open occult file for writing!");
    }
    virtual
    ~McmcWriter()
    {
      paramFile_.close();
      occFile_.close();
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
