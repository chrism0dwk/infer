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
#include <iterator>
#include <utility>
#include <string>
#include <limits>
#include <iostream>
#include <fstream>
#include <cassert>

#include "types.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"

using namespace std;

#define POSINF ( numeric_limits<double>::infinity() )

namespace EpiRisk
{

  //! Population data management class
  /*! The Population class provides a framework for
   * managing epidemic data within populations.
   */
  template<typename Covars>
    class Population
    {

    public:

      typedef Individual<Covars> Individual;
      typedef vector<Individual> PopulationContainer;
      typedef typename PopulationContainer::iterator iterator;
      typedef typename PopulationContainer::const_iterator const_iterator;

      struct PopulationIndexCmp
      {
        bool
        operator()(const iterator& lhs, const iterator& rhs) const
        {
          return lhs->getI() < rhs->getI();
        }
      };

      typedef multiset<iterator, PopulationIndexCmp> PopulationIndex;
      typedef map<string, iterator> IdMap;

      class IndexIterator : public std::iterator<bidirectional_iterator_tag, const_iterator>
      {
         typename PopulationIndex::iterator p_;
      public:
         IndexIterator(const typename PopulationIndex::iterator& it) :p_(it) {}
         IndexIterator& operator++() {++p_;return *this;}
         IndexIterator& operator--() {--p_;return *this;}
         bool operator==(const IndexIterator& rhs) {return p_==rhs.p_;}
         bool operator!=(const IndexIterator& rhs) {return p_!=rhs.p_;}
         Individual& operator*() {return **p_;}
         const Individual* operator->() {return &(**p_);}
      };

      // Ctor & Dtor
      Population();
      virtual
      ~Population();
      void
      resetEventTimes();

      // Data management methods
      //! Sets the population observation time
      void
      setObsTime(const double obsTime)
      {
        obsTime_ = obsTime;
      }
      //! Returns the population observation time
      double
      getObsTime() const
      {
        return obsTime_;
      }

      /*! \brief Imports population data.
       *
       * Use this method to import population data.
       * @param popDataImporter is derived from DataImporter, and provides
       * a next() method returning a class Covars.
       */
      void
      importPopData(DataImporter<Covars>& popDataImporter);
      /*! \brief Imports epidemic data.
       *
       * Use this method *after* importPopData to import epidemic data.
       * @param epiDataImporter is derives from DataImporter, and provides
       * a next() method returning an Events struct.
       */
      void
      importEpiData(DataImporter<Events>& epiDataImporter);
      //! Imports contact tracing XML
      void
      importContactTracing(const string filename);
      /*! Adds an occult infection
       * @param susceptibleIndex the position of the susceptible to add in the susceptibles index.
       */
      void
      addOccult(const size_t susceptibleIndex);
      /*! Deletes an occult infection
       * @param occultIndex the position of the occult to delete in the occult index.
       */
      void
      delOccult(const size_t occultIndex);
      //! Deletes all occult infections
      void
      resetOccults();
      void
      clear(); // Clears all infected individuals

      // Data access
      /*! Returns a const_iterator to an individual
       * @param pos the position of the individual in the population vector.
       */
      const_iterator
      operator[](const size_t pos) const;
      /// Returns a const_iterator to the beginning of the population vector.
      const_iterator
      begin() const;
      /// Returns a const_iterator to just past the end of the population vector.
      const_iterator
      end() const;
      /// Returns the size of the population.
      size_t
      size() const;
      /// Returns the current number of susceptibles.
      size_t
      numSusceptible() const;
      /// Returns the current number of infectives (including occults)
      size_t
      numInfected() const;
      /*! \brief Creates an infection.
       *
       * This method effectively infects a susceptible.  Note that this should be a
       * *known* infection, and *not* an occult infection.
       * @param index the position of the infection to add in the susceptible index.
       * @param I the infection time
       * @param N the notification time
       * @param R the removal time
       */
      void
      addInfec(size_t index, eventTime_t I, eventTime_t N = POSINF,
          eventTime_t R = POSINF);
      /*! \brief Deletes an infection
       *
       * This method deletes a *known* infection.
       * @param index the position of the infection in the infectives index.
       */
      void
      delInfec(size_t index);

      // Time functions
      /*! /brief Moves an infection time
       *
       * This function moves an infection time.
       * @param index the position of the infection time to move in the infectives index.
       * @param newTime the infection time to move to.
       */
      void
      moveInfectionTime(const size_t index, const double newTime);
      double
      exposureI(const size_t i, const size_t j) const; // Time for which j is exposed to infected i
      double
      exposureIBeforeCT(const size_t i, const size_t j) const;
      double
      ITimeBeforeCT(const size_t i) const;
      double
      exposureN(const size_t i, const size_t j) const; // Time for which j is exposed to notified i
      double
      ITime(const size_t i) const; // Time for whichtemplate<typename Individual> i was infective
      double
      NTime(const size_t i) const; // Time for which i was notified
      double
      STime(const size_t i) const; // Time for which i was susceptible
      IndexIterator
      I1() const
      {
        return IndexIterator(infectives_.begin());
      } // Returns iterator to I1

    private:
      // Private data
      PopulationContainer individuals_;
      PopulationIndex infectives_;
      PopulationIndex susceptibles_;
      IdMap idIndex_;

      double obsTime_;
      size_t knownInfections_;

      // Private methods
      void
      initContactTracing(const char* const );
      void
      updateInfecMethod();

    };

  //////////////////////////////////////////////////////
  ///////////////////// Implementation /////////////////
  //////////////////////////////////////////////////////

  template<typename Covars>
    Population<Covars>::Population() :
      obsTime_(POSINF), knownInfections_(0)
    {

    }

  template<typename Covars>
    Population<Covars>::~Population()
    {

    }

  template<typename Covars>
    void
    Population<Covars>::resetEventTimes()
    {
      for (size_t i = 0; i < individuals_.size(); ++i)
        {
          individuals_[i].I = obsTime_;
          individuals_[i].N = obsTime_;
          individuals_[i].R = obsTime_;
        }
    }

  template<typename Covars>
    void
    Population<Covars>::importPopData(DataImporter<Covars>& popDataImporter)
    {
      typename DataImporter<Covars>::Record record;
      popDataImporter.reset(); // Make sure we're at the beginning of the file

      try
        {
          while (1)
            {
              record = popDataImporter.next();
              individuals_.push_back(Individual(record.id, record.data));
            }
        }
      catch (fileEOF& e)
        {
          // Build indices
          idIndex_.clear();
          susceptibles_.clear();
          infectives_.clear();
          pair<typename IdMap::iterator, bool> rv;
          iterator iter = individuals_.begin();
          while (iter != individuals_.end())
            {
              // Add to id index
              rv = idIndex_.insert(typename IdMap::value_type(iter->getId(),
                  iter));
              if (rv.second == false)
                throw data_exception(
                    "Duplicate id found in population dataset!");

              // Add to susceptibles
              susceptibles_.insert(iter);
              iter++;
            }
        }
    }

  template<typename Covars>
    void
    Population<Covars>::importEpiData(DataImporter<Events>& epiDataImporter)
    {
      DataImporter<Events>::Record record;
      epiDataImporter.reset();
      try
        {
          while (1)
            {
              record = epiDataImporter.next();
              typename IdMap::iterator rv = idIndex_.find(record.id);
              if (rv == idIndex_.end())
                throw parse_exception(
                    "Key in epidemic data not found in population data");

              susceptibles_.erase(find(susceptibles_.begin(),
                  susceptibles_.end(), rv->second));
              rv->second->setEvents(record.data);
              infectives_.insert(rv->second);

            }
        }
      catch (fileEOF& e)
        {
          return;
        }
    }

  template<typename Covars>
    typename Population<Covars>::const_iterator
    Population<Covars>::operator[](const size_t pos) const
    {
      // Returns a pointer (looked up in individuals_) to an individual
      assert(pos < individuals_.size()); // Range check
      return individuals_.begin() + pos;
    }

  template<typename Covars>
    typename Population<Covars>::const_iterator
    Population<Covars>::begin() const
    {
      return individuals_.begin();
    }

  template<typename Covars>
    typename Population<Covars>::const_iterator
    Population<Covars>::end() const
    {
      return individuals_.end();
    }

  template<typename Covars>
    size_t
    Population<Covars>::size() const
    {
      return individuals_.size();
    }

  template<typename Covars>
    size_t
    Population<Covars>::numInfected() const
    {
      return infectives_.size();
    }

  template<typename Covars>
    size_t
    Population<Covars>::numSusceptible() const
    {
      return susceptibles_.size();
    }

  template<typename Covars>
    void
    Population<Covars>::importContactTracing(const string filename)
    {
      // Function associates infections with CT data
      SAXContactParse(filename, individuals_);
      updateInfecMethod();
    }

  template<typename Covars>
    void
    Population<Covars>::updateInfecMethod()
    {
      // Goes through infectives and finds if infected
      // by a contact or not.

      for (typename PopulationIndex::iterator iter = infectives_.begin(); iter
          != infectives_.end(); iter++)
        {
          (*infectives_)->updateInfecByContact();
        }
    }

  template<typename Covars>
    void
    Population<Covars>::addInfec(size_t index, eventTime_t I, eventTime_t N,
        eventTime_t R)
    {
      assert(index < susceptibles_.size());

      typename PopulationIndex::iterator toAdd = susceptibles_.begin();
      advance(toAdd, index);

      (*toAdd)->setI(I);
      (*toAdd)->setStatus(INFECTED);

      infectives_.insert(*toAdd);
      susceptibles_.erase(toAdd);
    }

  template<typename Covars>
    void
    Population<Covars>::delInfec(size_t index)
    {
      assert(index < infectives_.size());

      typename PopulationIndex::iterator toRemove = infectives_.begin();
      advance(toRemove, index);

      if ((*toRemove)->known)
        throw logic_error("Attempt to delete a known infection");

      (*toRemove)->setStatus(SUSCEPTIBLE);
      (*toRemove)->setI(POSINF);

      susceptibles_.insert(*toRemove);
      infectives_.erase(toRemove);
    }

  template<typename Covars>
    void
    Population<Covars>::moveInfectionTime(const size_t index,
        const double newTime)
    {
      typename PopulationIndex::iterator it = infectives_.begin();
      advance(it, index);
      typename Population::iterator tmp = *it;
      infectives_.erase(it);
      tmp->setI(newTime);
      infectives_.insert(tmp);
    }

  template<typename Covars>
    double
    Population<Covars>::exposureI(size_t i, size_t j) const
    {
      // NB: This gives time that susceptible j is exposed to infective i before becoming infected
      assert(i < size() and j < size());
      return min(individuals_[i].N, individuals_[j].I) - min(individuals_[i].I,
          individuals_[j].I);
    }

  template<typename Covars>
    double
    Population<Covars>::exposureIBeforeCT(size_t i, size_t j) const
    {
      // Returns time that susceptible j is exposed to infective i before
      // either its contact tracing started or it got infected (the latter in the
      // unlikely event that it was infected before the contact tracing started)

      double stopTime;
      double startTime;
      assert(i < size() and j < size());
      double earliestContactStart = min(individuals_[i].getContactStart(),
          individuals_[j].getContactStart());

      stopTime = min(individuals_[i].getN(), individuals_[j].getI());
      stopTime = min(earliestContactStart, stopTime);

      startTime = min(individuals_[j].getI(), individuals_[i].getI());
      startTime = min(earliestContactStart, stopTime);

      return stopTime - startTime;
    }

  template<typename Covars>
    double
    Population<Covars>::ITimeBeforeCT(size_t i) const
    {
      // Returns the amount of time between I and start of CT window
      // Non-neg if CTstart > I, 0 otherwise
      assert(i < size());
      double iTime = individuals_[i].getContactStart() - individuals_[i].getI();

      if (iTime > 0)
        return iTime;
      else
        return 0.0;
    }

  template<typename Covars>
    double
    Population<Covars>::exposureN(size_t i, size_t j) const
    {
      // NB: This gives time that susceptible j is exposed to notified i before becoming infected
      assert(i < size() and j < size());
      return min(individuals_[i].getR(), individuals_[j].getI()) - min(
          individuals_[j].getI(), individuals_[i].getN());
    }

  template<typename Covars>
    double
    Population<Covars>::ITime(size_t i) const
    {
      // NB: Gives the time for which i was infectious but not notified
      assert(i < size());
      return individuals_[i].getN() - individuals_[i].getI();
    }

  template<typename Covars>
    double
    Population<Covars>::NTime(size_t i) const
    {
      // NB: Gives the time for which i was notified
      assert(i < size());
      return individuals_[i].getR() - individuals_[i].getN();
    }

  template<typename Covars>
    double
    Population<Covars>::STime(size_t i) const
    {
      // NB: Gives the time for which i was susceptible
      assert(i < size());
      return individuals_[i].getI() - individuals_[I1].getI();
    }

}
#endif /* POPULATION_HPP_ */
