/*************************************************************************
 *  ./src/Framework/SpatPointPop.hpp
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
#include <algorithm>
#include <utility>
#include <string>
#include <limits>
#include <iostream>
#include <fstream>
#include <cassert>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/composite_key.hpp>

#include "types.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"
#include "EpiRiskException.hpp"
#include "stlStrTok.hpp"

using namespace std;
using boost::multi_index_container;
using namespace boost::multi_index;

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
      typedef EpiRisk::Individual<Covars> Individual;

    private:
      // Index tags
      struct bySeq {};
      struct byI {};
      struct byId {};
      struct byN {};
      struct byDiseaseStatus {};

      // Population multi_index_container
      typedef boost::multi_index_container<
          Individual,
          indexed_by<
            random_access< tag<bySeq> >,
            hashed_unique< tag<byId>, const_mem_fun<Individual,string,&Individual::getId> >,
            ordered_non_unique<tag<byI>, const_mem_fun<Individual,double,&Individual::getI> >
          >
      >  PopulationContainer;




      // Population Indices
      typedef typename PopulationContainer::template index<byI>::type InfectiveIndex;
      typedef typename PopulationContainer::template index<bySeq>::type PopulationIndex;
      typedef typename PopulationContainer::template index<byId>::type IdIndex;

    public:
      // Iterators
      typedef typename PopulationIndex::const_iterator PopulationIterator;
      typedef typename InfectiveIndex::const_iterator InfectiveIterator;

    private: // Comparison functors
      struct cmpI
      {
        bool operator()(const double l,const double r) const {
          bool rv = l < r;
          return rv;}
      };

    // Occult index
      typedef boost::multi_index_container<
          InfectiveIterator,
          indexed_by< sequenced<> >
      > OccultList;


    public: // methods
      /// \name Constructor and destructor
      //@{
      Population();
      virtual
      ~Population();
      //@}


      /// \name Data management methods
      //@{
      //! Sets the population observation time
      void
      setObsTime(const double obsTime);
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
      //@}



      /// \name Population management methods
      //@{
      /*! \brief Creates an infection.
       *
       * This method effectively infects a susceptible.  Note that this should be a
       * *known* infection, and *not* an occult infection.
       * @param index the position of the infection to add in the susceptible index.
       * @param I the infection time
       * @param N the notification time
       * @param R the removal time
       */
      bool
      addInfec(size_t index, eventTime_t I, eventTime_t N = POSINF,
          eventTime_t R = POSINF);
      /*! \brief Deletes an infection
       *
       * This method deletes a *known* infection.
       * @param index the position of the infection in the infectives index.
       */
      bool
      delInfec(size_t index);

      /*! /brief Moves an infection time
       *
       * This function moves an infection time.
       * @param index the position of the infection time to move in the infectives index.
       * @param newTime the infection time to move to.
       */
      bool
      moveInfectionTime(const InfectiveIterator& it, const double newTime);
      /*! /brief Moves an infection time
       *
       * This function moves an infection time.
       * @param id the id of the infection time to move in the infectives index.
       * @param newTime the infection time to move to.
       */
      bool
      moveInfectionTime(const std::string id, const double newTime);
      /*! /brief Updates I, N, and R event times
       *
       * @param individual a reference to the individual to update
       * @param events an Events struct containing the new times
       * @return whether the operation was successful
       */
      bool
      updateEvents(const Individual& individual,Events events);
      /*! Adds an occult infection
       * @param susceptibleIndex the position of the susceptible to add in the susceptibles index.
       */
      //! Clears all infected individuals
      void
      clearInfections()
      {
        for(PopulationIterator it = population_.begin();
            it != population_.end();
            it++)
          moveInfectionTime(asI(it),POSINF);
      }
      /*! \brief Creates a connection graph
       * \param predicate a functor whose operator() method returns true if two individuals are connected
       */
      template<class Predicate>
      void
      createConnectionGraph(Predicate predicate)
      {
        size_t counter = 0;
        for(PopulationIterator it = population_.begin();
            it != population_.end();
            it++)
          {
            cerr << counter << endl; ++ counter;
            typename Individual::ConnectionList tmp;
            for(PopulationIterator jt = population_.begin();
                jt != it;
                ++jt)
              {
                if (predicate(*it,*jt)) {
                    const_cast<typename Individual::ConnectionList&>(it->getConnectionList()).push_back(&(*jt));
                    const_cast<typename Individual::ConnectionList&>(jt->getConnectionList()).push_back(&(*it));
                }
              }
          }
      }

      void
      loadConnectionGraph(const string filename)
      {
        ifstream confile;
        confile.open(filename.c_str(),ios::in);
        if(!confile.is_open()) {
            throw data_exception("Cannot open specified connections file");
        }

        string line;
        getline(confile,line); // Skip CSV header

        while(!confile.eof()) {
            getline(confile,line);
            std::vector<std::string> toks;
            stlStrTok(toks,line,",");
            if(toks.size() != 2) break;
            const_cast<typename Individual::ConnectionList&>(getById(toks[0]).getConnectionList()).push_back(&(getById(toks[1])));
        }

        confile.close();
      }
      //@}

      /// \name Data access methods
      //@{
      //! Returns the population observation time
      double
      getObsTime() const;
      /*! Returns a reference to a const individual
       * @param pos the position of the individual in the population vector.
       */
      const Individual&
      operator[](const size_t pos) const;
      /*! Returns a reference to a const individual
       * @param id the id of the individual
       * @return A reference to a (const) individual
       */
      const Individual&
      getById(const std::string id) const;
      /// Returns a const_iterator to the beginning of the population vector.
      PopulationIterator
      begin() const;
      /// Returns a const_iterator to just past the end of the population vector.
      PopulationIterator
      end() const;
      /// Returns the size of the population.
      size_t
      size() const;
      /// Returns the current number of susceptibles.
      size_t
      numSusceptible();
      /// Returns the current number of infectives (including occults)
      size_t
      numInfected();
      /// Returns a const reference to I1
      const Individual&
      I1() const;
      /// Returns an iterator to the beginning of the infectives
      InfectiveIterator
      infecBegin() const;
      /// Returns an iterator to one past the end of the infectives
      InfectiveIterator
      infecEnd() const;
      /// Returns an iterator to the beginning of the occults
      InfectiveIterator
      occultBegin() const;
      /// Returns an iterator to one past the end of the occults
      InfectiveIterator
      occultEnd() const;
      /// Returns an iterator to the infective with infection time <= I
      InfectiveIterator
      infecUpperBound(const double I);
      /// Returns an iterator to the infectives with infection time < I
      InfectiveIterator
      infecLowerBound(const double I);
      /// Converts an iterator to a PopulationIterator
      template<typename IteratorType>
      PopulationIterator
      asPop(const IteratorType& it)
      {
        return population_.project<bySeq>(it);
      }
      /// Converts an iterator to an InfectiveIterator
      template<typename IteratorType>
      InfectiveIterator
      asI(const IteratorType& it)
      {
        return population_.project<byI>(it);
      }
      //@}


      /// \name Debug methods
      //@{
      /// Dumps the infection times to stderr
      void
      dumpInfected()
      {
        cerr.precision(15);
        InfectiveIndex& iIndex = population_.get<byI>();
        typename InfectiveIndex::iterator it = iIndex.begin();
        while(it->getI() <= obsTime_) {
            cerr << it->getId() << "\t"
                << it->getI() << "\t"
                << it->getN() << "\t"
                << it->getR() << endl;
            it++;
        }
      }
      /// Dumps the entire population to stderr
      void
      dumpPopulation()
      {
        for(PopulationIterator popIndex = population_.begin();
            popIndex != population_.end();
            popIndex++)
          {
            cerr << popIndex->getId() << "\t"
                << popIndex->getI() << "\t"
                << popIndex->getN() << "\t"
                << popIndex->getR() << endl;
          }
      }
      //@}


      /// \name Time methods
      //@{
      /// Time for which j is exposed to i
      double
      exposureI(const size_t i, const size_t j) const;
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
      //@}


    private:
      // Private data
      PopulationContainer population_;
      OccultList occultList_;

      InfectiveIndex& infIndex_;
      PopulationIndex& seqIndex_;
      IdIndex& idIndex_;

      double obsTime_;
      size_t knownInfections_;

      // Private methods
      void
      initContactTracing(const char* const );
      void
      updateInfecMethod();

      struct modifyEvents
      {
        modifyEvents(Events& events) : events_(events) {}
        void operator()(Individual& i)
        {
          i.setEvents(events_);
        }
      private:
        Events& events_;
      };

      struct modifyI
      {
        modifyI(double newI) : newI_(newI) {}
        void operator()(Individual& i)
        {
          i.setI(newI_);
        }
      private:
        double newI_;
      };

      struct modifyStatus
      {
        modifyStatus(InfecStatus_e newStatus) : newStatus(newStatus) {}
        void operator()(Individual& i)
        {
          i.setStatus(newStatus);
        }
      private:
        InfecStatus_e newStatus;
      };

    };

  //////////////////////////////////////////////////////
  ///////////////////// Implementation /////////////////
  //////////////////////////////////////////////////////

  template<typename Covars>
    Population<Covars>::Population() :
      obsTime_(POSINF), knownInfections_(0),
      idIndex_(population_.get<byId>()),
      infIndex_(population_.get<byI>()),
      seqIndex_(population_.get<bySeq>())
    {

    }

  template<typename Covars>
    Population<Covars>::~Population()
    {

    }

  template<typename Covars>
  void
  Population<Covars>::setObsTime(const double obsTime)
  {
    obsTime_ = obsTime;
  }

  template<typename Covars>
  double
  Population<Covars>::getObsTime() const
  {
    return obsTime_;
  }

  template<typename Covars>
    void
    Population<Covars>::importPopData(DataImporter<Covars>& popDataImporter)
    {
      typename DataImporter<Covars>::Record record;
      popDataImporter.open();

      try
        {
          InfectiveIndex& index = population_.get<byI>();
          while (1)
            {
              record = popDataImporter.next();
              index.insert( Individual(record.id, record.data) );
            }
        }
      catch (fileEOF& e)
        {
          return;
        }

      popDataImporter.close();
    }

  template<typename Covars>
    void
    Population<Covars>::importEpiData(DataImporter<Events>& epiDataImporter)
    {
      DataImporter<Events>::Record record;
      epiDataImporter.open();
      try
        {
          IdIndex& idIndex = population_.get<byId>();
          while (1)
            {
              record = epiDataImporter.next();
              typename IdIndex::iterator ref = idIndex.find(record.id);
              if (ref == idIndex.end())
                throw parse_exception(
                    "Key in epidemic data not found in population data");
              Events oldEvents = ref->getEvents();

              // Check data integrity
              if (record.data.N > record.data.R) {
                  cerr << "Individual " << record.id << " has N > R.  Setting N = R\n";
                  record.data.N = record.data.R;
              }

              idIndex.modify(ref,modifyEvents(record.data),modifyEvents(oldEvents));
            }
        }
      catch (fileEOF& e)
        {
          return;
        }

      epiDataImporter.close();
    }

  template<typename Covars>
    void
    Population<Covars>::importContactTracing(const string filename)
    {
      // Function associates infections with CT data
      SAXContactParse(filename, population_);
      updateInfecMethod();
    }

  template<typename Covars>
    const typename Population<Covars>::Individual&
    Population<Covars>::operator[](const size_t pos) const
    {
      // Returns a const reference to an individual
      assert(pos < population_.size()); // Range check
      return population_[pos];
    }

  template<typename Covars>
  const typename Population<Covars>::Individual&
  Population<Covars>::getById(const std::string id) const
  {
    typename IdIndex::const_iterator it = idIndex_.find(id);
    if(it == idIndex_.end())
      {
         std::stringstream ss; ss << "ID '" << id << "' not found in population!";
         throw runtime_error(ss.str().c_str());
      }
    return *it;
  }

  template<typename Covars>
    typename Population<Covars>::PopulationIterator
    Population<Covars>::begin() const
    {
      return population_.begin();
    }

  template<typename Covars>
    typename Population<Covars>::PopulationIterator
    Population<Covars>::end() const
    {
      return population_.end();
    }

  template<typename Covars>
    size_t
    Population<Covars>::size() const
    {
      return population_.size();
    }

  template<typename Covars>
    size_t
    Population<Covars>::numInfected()
    {
      return distance(infIndex_.begin(),infIndex_.lower_bound(obsTime_));
    }

  template<typename Covars>
    size_t
    Population<Covars>::numSusceptible()
    {
      return distance(infIndex_.upper_bound(obsTime_),infIndex_.end());
    }

  template<typename Covars>
    void
    Population<Covars>::updateInfecMethod()
    {
      // Goes through infectives and finds if infected
      // by a contact or not.

      for (typename InfectiveIndex::iterator iter = infIndex_.begin();
           iter != infIndex_.end();
           iter++)
        {
          iter->updateInfecByContact();
        }
    }

  template<typename Covars>
    bool
    Population<Covars>::addInfec(size_t index, eventTime_t I, eventTime_t N,
        eventTime_t R)
    {
      assert(index < numSusceptible());

      typename InfectiveIndex::iterator toAdd = infIndex_.upper_bound(obsTime_);
      advance(toAdd, index);

      Events newEvents;
      newEvents.I = I;
      newEvents.N = N;
      newEvents.R = R;

      Events oldEvents = toAdd->getEvents();

      return  infIndex_.modify(toAdd,modifyEvents(newEvents),modifyEvents(oldEvents)) and
              infIndex_.modify(toAdd,modifyStatus(INFECTED),modifyStatus(SUSCEPTIBLE));
    }

  template<typename Covars>
    bool
    Population<Covars>::delInfec(size_t index)
    {
      assert(index < numInfected());

      typename InfectiveIndex::iterator toRemove = infIndex_.begin();
      advance(toRemove, index);

      if (toRemove->known)
        throw logic_error("Attempt to delete a known infection");

      Events newEvents;
      newEvents.I = POSINF;
      newEvents.N = POSINF;
      newEvents.R = POSINF;

      Events oldEvents = toRemove->getEvents();

      return infIndex_.modify(toRemove,modifyEvents(newEvents),modifyEvents(oldEvents)) and
             infIndex_.modify(toRemove,modifyStatus(SUSCEPTIBLE),modifyStatus(INFECTED));
    }

  template<typename Covars>
    bool
    Population<Covars>::moveInfectionTime(const InfectiveIterator& it,
        const double newTime)
    {
      double oldTime = it->getI();
      bool  rv = infIndex_.modify(it,modifyI(newTime),modifyI(oldTime));
      if (rv == false) throw logic_error("Failed to modify infection time!!");
      return rv;

    }

  template<typename Covars>
    bool
    Population<Covars>::moveInfectionTime(const std::string id, const double newTime)
    {
      typename IdIndex::iterator idIter = idIndex_.find(id);
      if (idIter == idIndex_.end()) throw data_exception("Id not found");
      typename InfectiveIndex::iterator infIter = population_.project<byI>(idIter);

      return moveInfectionTime(infIter, newTime);
    }

  template<typename Covars>
  bool
  Population<Covars>::updateEvents(const Individual& individual,Events events)
  {
    typename InfectiveIndex::iterator ref = infIndex_.iterator_to(individual);
    Events oldEvents = individual.getEvents();
    return infIndex_.modify(ref,modifyEvents(events),modifyEvents(oldEvents));
  }

  template<typename Covars>
    double
    Population<Covars>::exposureI(size_t i, size_t j) const
    {
      // NB: This gives time that susceptible j is exposed to infective i before becoming infected
      assert(i < size() and j < size());
      return min(population_[i].N, population_[j].I) - min(population_[i].I,
          population_[j].I);
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
      double earliestContactStart = min(population_[i].getContactStart(),
          population_[j].getContactStart());

      stopTime = min(population_[i].getN(), population_[j].getI());
      stopTime = min(earliestContactStart, stopTime);

      startTime = min(population_[j].getI(), population_[i].getI());
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
      double iTime = population_[i].getContactStart() - population_[i].getI();

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
      return min(population_[i].getR(), population_[j].getI()) - min(
          population_[j].getI(), population_[i].getN());
    }

  template<typename Covars>
    double
    Population<Covars>::ITime(size_t i) const
    {
      // NB: Gives the time for which i was infectious but not notified
      assert(i < size());
      return population_[i].getN() - population_[i].getI();
    }

  template<typename Covars>
    double
    Population<Covars>::NTime(size_t i) const
    {
      // NB: Gives the time for which i was notified
      assert(i < size());
      return population_[i].getR() - population_[i].getN();
    }

  template<typename Covars>
    double
    Population<Covars>::STime(size_t i) const
    {
      // NB: Gives the time for which i was susceptible
      assert(i < size());
      return population_[i].getI() - population_[0].getI();
    }

  template<typename Covars>
  const typename Population<Covars>::Individual&
  Population<Covars>::I1() const
  {
    return *(infIndex_.begin());
  }

  template<typename Covars>
  typename Population<Covars>::InfectiveIterator
  Population<Covars>::infecBegin() const
  {
    return infIndex_.begin();
  }

  template<typename Covars>
  typename Population<Covars>::InfectiveIterator
  Population<Covars>::infecEnd() const
  {
    typename Population<Covars>::InfectiveIterator it = infIndex_.upper_bound(obsTime_);
    return it;
  }

  template<typename Covars>
  typename Population<Covars>::InfectiveIterator
  Population<Covars>::infecUpperBound(const double I)
  {
    return infIndex_.upper_bound(min(I,obsTime_));
  }

  template<typename Covars>
  typename Population<Covars>::InfectiveIterator
  Population<Covars>::infecLowerBound(const double I)
  {
    return infIndex_.lower_bound(I);
  }

}  // Namespace EpiRisk
#endif /* POPULATION_HPP_ */
