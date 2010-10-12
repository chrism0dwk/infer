/***************************************************************************
 *   Copyright (C) 2010 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

/*
 * Population.hpp
 *
 *  Created on: 11 Dec 2009
 *      Author: Chris Jewell
 */

#ifndef POPULATION_HPP_
#define POPULATION_HPP_

#include <vector>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <gsl/gsl_math.h>

#include "types.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"

using namespace std;

namespace EpiRisk
{

  template<typename Indiv>
    class Population
    {

    private:

      // Private data
      int rv;

      // Private methods
      void
      initContactTracing(const char* const );
      void
      updateInfecMethod();

    public:

      typedef vector<Indiv> PopulationContainer;
      typedef typename PopulationContainer::iterator iterator;
      typedef multiset<iterator> PopulationIndex;
      typedef Indiv indivtype;
      typedef map<string,Indiv*> IdMap;

      /* data */

      PopulationContainer individuals_;
      PopulationIndex infectives_;
      PopulationIndex susceptibles_;
      IdMap idIndex_;


      double obsTime_;
      size_t knownInfections_;

      // Ctor & Dtor
      Population();
      virtual
      ~Population();
      void
      resetEventTimes();

      // Data management methods
      void
      setObsTime(const double obsTime)
      {
        obsTime_ = obsTime;
      }
      double
      getObsTime() const
      {
        return obsTime_;
      }
      void
      importPopData(DataImporter<typename Indiv::CovarsType>& popDataImporter);
      void
      importEpiData(DataImporter<Events>& epiDataImporter);
      void
      importContactTracing(const string filename);
      int
      addInfec(Ilabel_t, eventTime_t, eventTime_t, eventTime_t);
      int
      delInfec(Ipos_t);
      void
      resetOccults(); // Deletes all occults
      void
      clear(); // Clears all infected individuals

      // Data access
      Indiv*
      operator[](const size_t pos);
      iterator
      begin();
      iterator
      end();
      size_t
      size();

      // Time functions
      double
      exposureI(Ipos_t, Ipos_t); // Time for which j is exposed to infected i
      double
      exposureIBeforeCT(Ipos_t, Ipos_t);
      double
      ITimeBeforeCT(Ipos_t);
      double
      exposureN(Ipos_t, Ipos_t); // Time for which j is exposed to notified i
      double
      ITime(Ipos_t); // Time for whichtemplate<typename Individual> i was infective
      double
      NTime(Ipos_t); // Time for which i was notified
      double
      STime(Ipos_t); // Time for which i was susceptible
      iterator
      I1()
      {
        return infectives_.first();
      }
      ; // Returns iterator to I1
      iterator
      I2()
      {
        return infectives_.first()++;
      }
      ; // Returns iterator to I2


    };

  //////////////////////////////////////////////////////
  ///////////////////// Implementation /////////////////
  //////////////////////////////////////////////////////

  template<typename Indiv>
    Population<Indiv>::Population() :
      obsTime_(GSL_POSINF), knownInfections_(0)
    {

    }

  template<typename Indiv>
    Population<Indiv>::~Population()
    {

    }

  template<typename Indiv>
    void
    Population<Indiv>::resetEventTimes()
    {
      for (size_t i = 0; i < individuals_.size(); ++i)
        {
          individuals_[i].I = obsTime_;
          individuals_[i].N = obsTime_;
          individuals_[i].R = obsTime_;
        }
    }

  template<typename Indiv>
    void
    Population<Indiv>::importPopData(DataImporter<typename Indiv::CovarsType>& popDataImporter)
    {
      typename DataImporter<typename Indiv::CovarsType>::Record record;
      popDataImporter.reset(); // Make sure we're at the beginning of the file

      try
        {
          while (1)
            {
              record = popDataImporter.next();
              individuals_.push_back(Indiv(record.id,record.data));
            }
        }
      catch (fileEOF& e)
        {
          // Build id index
          idIndex_.clear();
          pair<typename map<string,Indiv*>::iterator, bool> rv;
          typename PopulationContainer::iterator iter = individuals_.begin();
          while(iter != individuals_.end())
            {
              rv = idIndex_.insert(typename IdMap::value_type(iter->getId(),&(*iter)));
              if(rv.second == false) throw data_exception("Duplicate id found in population dataset!");
              iter++;
            }
        }
    }

  template<typename Indiv>
    void
    Population<Indiv>::importEpiData(DataImporter<Events>& epiDataImporter)
    {
      DataImporter<Events>::Record record;
      epiDataImporter.reset();
      try
      {
          while(1)
            {
              record = epiDataImporter.next();
              idIndex_[record.id]->setEvents(record.data);
            }
      }
      catch (fileEOF& e)
      {
          return;
      }
    }

  template<typename Indiv>
    Indiv*
    Population<Indiv>::operator[](const size_t pos)
    {
      // Returns a pointer (looked up in individuals) to an individual
      assert(pos < individuals_.size()); // Range check
      return &individuals_[pos];
    }

  template<typename Indiv>
    typename Population<Indiv>::iterator
    Population<Indiv>::begin()
    {
      return individuals_.begin();
    }

  template<typename Indiv>
    typename Population<Indiv>::iterator
    Population<Indiv>::end()
    {
      return individuals_.end();
    }

  template<typename Indiv>
    size_t
    Population<Indiv>::size()
    {
      return individuals_.size();
    }

  template<typename Indiv>
    void
    Population<Indiv>::importContactTracing(const string filename)
    {
      // Function associates infections with CT data
      SAXContactParse(filename, individuals_);
      updateInfecMethod();
    }

  template<typename Indiv>
    void
    Population<Indiv>::updateInfecMethod()
    {
      // Goes through infectives and finds if infected
      // by a contact or not.

      for (typename PopulationIndex::iterator iter = infectives_.begin(); iter
          != infectives_.end(); iter++)
        {
          (*infectives_)->updateInfecByContact();
        }
    }

  template<typename Indiv>
    int
    Population<Indiv>::addInfec(Spos_t susc_pos, eventTime_t thisI,
        eventTime_t thisN, eventTime_t thisR)
    {

      susceptibles_.at(susc_pos)->I = thisI;
      susceptibles_.at(susc_pos)->status = Indiv::INFECTED;
      infectives_.push_back(susceptibles_.at(susc_pos));
      susceptibles_.erase(susceptibles_.begin() + susc_pos);
      return (0);
    }

  template<typename Indiv>
    int
    Population<Indiv>::delInfec(Ipos_t infec_pos)
    {
      if (infectives_.at(infec_pos)->known == 1)
        throw logic_error("Deleting known infection");

      susceptibles_.push_back(infectives_.at(infec_pos));
      susceptibles_.back()->status = Indiv::SUSCEPTIBLE;
      susceptibles_.back()->I = susceptibles_.back()->N;
      infectives_.erase(infectives_.begin() + infec_pos);
      return (0);
    }

  template<typename Indiv>
    double
    Population<Indiv>::exposureI(Ipos_t i, Ipos_t j)
    {
      // NB: This gives time that susceptible j is exposed to infective i before becoming infected
      return min(individuals_[i].N, individuals_[j].I) - min(individuals_[i].I,
          individuals_[j].I);
    }

  template<typename Indiv>
    double
    Population<Indiv>::exposureIBeforeCT(Ipos_t i, Ipos_t j)
    {
      // Returns time that susceptible j is exposed to infective i before
      // either its contact tracing started or it got infected (the latter in the
      // unlikely event that it was infected before the contact tracing started)

      double stopTime;
      double startTime;
      double earliestContactStart = min(individuals_[i].getContactStart(),
          individuals_[j].getContactStart());

      stopTime = min(individuals_[i].getN(), individuals_[j].getI());
      stopTime = min(earliestContactStart, stopTime);

      startTime = min(individuals_[j].getI(), individuals_[i].getI());
      startTime = min(earliestContactStart, stopTime);

      return stopTime - startTime;
    }

  template<typename Indiv>
    double
    Population<Indiv>::ITimeBeforeCT(Ipos_t i)
    {
      // Returns the amount of time between I and start of CT window
      // Non-neg if CTstart > I, 0 otherwise

      double iTime = individuals_.at(i).getContactStart()
          - individuals_[i].getI();

      if (iTime > 0)
        return iTime;
      else
        return 0.0;
    }

  template<typename Indiv>
    double
    Population<Indiv>::exposureN(Ipos_t i, Ipos_t j)
    {
      // NB: This gives time that susceptible j is exposed to notified i before becoming infected
      return min(individuals_[i].getR(), individuals_[j].getI()) - min(
          individuals_[j].getI(), individuals_[i].getN());
    }

  template<typename Indiv>
    double
    Population<Indiv>::ITime(Ipos_t i)
    {
      // NB: Gives the time for which i was infectious but not notified
      return individuals_.at(i).getN() - individuals_[i].getI();
    }

  template<typename Indiv>
    double
    Population<Indiv>::NTime(Ipos_t i)
    {
      // NB: Gives the time for which i was notified
      return individuals_[i].getR() - individuals_[i].getN();
    }

  template<typename Indiv>
    double
    Population<Indiv>::STime(Ipos_t i)
    {
      // NB: Gives the time for which i was susceptible
      return individuals_[i].getI() - individuals_[I1].getI();
    }

}
#endif /* POPULATION_HPP_ */
