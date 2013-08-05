/*************************************************************************
 *  ./src/sim/GillespieSim.hpp
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
 * GillespieSim.hpp
 *
 *  Created on: Jun 14, 2011
 *      Author: stsiab
 */

#ifndef GILLESPIESIM_HPP_
#define GILLESPIESIM_HPP_

#include <cstdlib>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <tr1/memory>
#include <gsl/gsl_cdf.h>

#include "types.hpp"
#include "Random.hpp"
#include "Model.hpp"

namespace EpiRisk
{

  using std::tr1::shared_ptr;

  template<typename Model>
    class GillespieSim
    {
    public:
      GillespieSim(Model& model, Random& random);
      virtual
      ~GillespieSim();

      void
      setMaxTime(const double maxTime = POSINF);
      double
      getMaxTime() const;
      void
      simulate(bool simCensoredEvents = false);
      void
      dumpPressureCDF() const;
      void
      serialize(const std::string filename, const bool censored = true) const;

    private:

      typedef map<double, const typename Model::Individual*> PressureCDF;

      class Event
      {
      public:
        Event(const typename Model::Individual* individual, const double time) :
          individual_(individual), time_(time)
        {
        }
        virtual
        ~Event()
        {
        }
        virtual
        double
        getTime() const
        {
          return time_;
        }
        virtual const typename Model::Individual*
        getIndividual() const
        {
          return individual_;
        }
        virtual
        bool
        operator<(Event& rhs)
        {
          return this->time_ < rhs.getTime();
        }

      protected:
        const typename Model::Individual* individual_;
        double time_;
      };

      class Infection : public Event
      {
      public:
        Infection(typename PressureCDF::iterator pressure, const double time) :
          Event(pressure->second, time), pressure_(pressure)
        {
        }
        virtual const typename Model::Individual*
        getIndividual() const
        {
          return this->individual_;
        }
        virtual typename PressureCDF::iterator
        getPressureIterator() const
        {
          return pressure_;
        }
      private:
        typename PressureCDF::iterator pressure_;
      };

      class Notification : public Event
      {
      public:
        Notification(const typename Model::Individual* individual, double time) :
          Event(individual, time)
        {
        }
      };

      class Removal : public Event
      {
      public:
        Removal(const typename Model::Individual* individual, double time) :
          Event(individual, time)
        {
        }
      };

      class Ghost : public Event
      {
      public:
    	  Ghost(const typename Model::Individual* individual, const double time) :
    		  Event(individual, time) {};
      };

      struct EventQueueCmp {
        bool operator() (const shared_ptr<Event>& lhs, const shared_ptr<Event>& rhs) const
        {
          return lhs->getTime() < rhs->getTime();
        }
      };
      typedef set<shared_ptr<Event>, EventQueueCmp > EventQueue;
      Model& model_;
      Random& random_;
      double maxTime_;
      PressureCDF pressureCDF_;
      PressureCDF infectorCDF_;
      EventQueue eventQueue_;
      typename Model::PopulationType& population_;
      int numS_, numI_, numN_, numR_;
      size_t numInfecs_;

      void
      calcPressureCDF();
      bool
      isInfectious(const typename Model::Individual* individual, const double time);
      void
      checkPressureCDF() const;
      void
      simulateCensoredEvents();
      void
      initEventQueue();
      double
      beta_max() const;
      //typename PressureCDF::iterator sampleInfection(const PressureCDF cdf) const;

      void
      infect(Infection* infection);
      void
      notify(Notification* notification);
      void
      remove(Removal* removal);

    };

  template<typename Model>
    GillespieSim<Model>::GillespieSim(Model& model, Random& random) :
      model_(model), random_(random), population_(model_.getPopulation()),
          maxTime_(POSINF),numInfecs_(0)
    {

    }

  template<typename Model>
    GillespieSim<Model>::~GillespieSim()
    {

    }

  template<typename Model>
    void
    GillespieSim<Model>::setMaxTime(const double maxTime)
    {
      maxTime_ = maxTime;
    }

  template<typename Model>
    double
    GillespieSim<Model>::getMaxTime() const
    {
      return maxTime_;
    }

  template<typename Model>
    void
    GillespieSim<Model>::calcPressureCDF()
    {
      pressureCDF_.clear();
      double cumulativePressure = 0.0;

      typename Model::PopulationType::InfectiveIterator it = ++population_.infecEnd();

      for (; it != population_.infecPopEnd(); it++)
        {
          cumulativePressure += model_.instantPressureOn(it,
              model_.getObsTime());
          pressureCDF_.insert(pair<double, const typename Model::Individual*> (
              cumulativePressure, &(*it)));
        }
    }

  template<typename Model>
  	bool
  	GillespieSim<Model>::isInfectious(const typename Model::Individual* infectee, const double time)
  	{
	  infectorCDF_.clear();
	  double cumulativePressure = 0.0;

	  for(typename EventQueue::const_iterator i = eventQueue_.begin();
			  i != eventQueue_.end();
			  ++i)
	  {
		  Removal* removal = dynamic_cast<Removal*> (i->get());
		  if (removal)
		  {
			  const typename Model::Individual* infector = (*i)->getIndividual();
			  if (infector->getN() > time)
			  {
				  cumulativePressure += model_.beta(*infector, *infectee, time);
				  infectorCDF_.insert(make_pair(cumulativePressure, infector));
			  }
			  else
			  {
				  cumulativePressure += model_.betastar(*infector, *infectee, time);
			  }
		  }
	  }

	  typename PressureCDF::const_iterator chosen = infectorCDF_.lower_bound(random_.uniform(0.0, cumulativePressure));

	  if(chosen != infectorCDF_.end())
		  return random_.uniform() < model_.hFunction(*(chosen->second), time);
  	}

  template<typename Model>
    void
    GillespieSim<Model>::simulateCensoredEvents()
    {
      cerr << "Simulating censored events" << endl;
      typename Model::PopulationType::InfectiveIterator stop =
          population_.infecLowerBound(population_.getObsTime());
      for (typename Model::PopulationType::InfectiveIterator it =
          population_.infecBegin(); it != stop; it++)
        {
          if (it->getN() > population_.getObsTime())
            {
              cerr << "Simulating for individual " << it->getId() << endl;
              Events events;
              events.I = it->getI();
              events.N = events.I + model_.leftTruncatedItoN(random_, *it);
              events.R = events.N + model_.NtoR();

              population_.updateEvents(*it,events);
            }

          if (it->getN() < population_.getObsTime() and it->getR() > population_.getObsTime())
            {
              Events events = it->getEvents();
              double Rcan = it->getN() + model_.NtoR();
              // If R is going to be < obsTime, set removal to be tomorrow -- fudgy!
              events.R = Rcan > population_.getObsTime() ? Rcan : population_.getObsTime() + model_.NtoR();
              population_.updateEvents(*it, events);
            }

        }
    }

  template<typename Model>
    void
    GillespieSim<Model>::initEventQueue()
    {
      eventQueue_.clear();

      // Iterate over individuals infected before currentTime_
      // and add notification and removal events
      typename Model::PopulationType::InfectiveIterator stop =
          population_.infecLowerBound(population_.getObsTime());
      for (typename Model::PopulationType::InfectiveIterator it =
          population_.infecBegin(); it != stop; it++)
        {
          if (it->getN() > population_.getObsTime() && it->getN() < POSINF)
            {
              if(!eventQueue_.insert(shared_ptr<Notification> (new Notification(
										&(*it), it->getN()))).second) {
		eventQueue_.insert(shared_ptr<Notification> (new Notification(&(*it), it->getN()+0.00001)));
		cerr << "WARNING: Duplicate N event time!" << endl;;
	      }
            }

          if (it->getR() > population_.getObsTime() && it->getR() < POSINF)
            {
              if(!eventQueue_.insert(shared_ptr<Removal> (new Removal(&(*it),
								      it->getR()))).second) {
		eventQueue_.insert(shared_ptr<Removal> (new Removal(&(*it), it->getR()+0.00001)));
		cerr << "WARNING: Duplicate R event time!" << endl;
	      }
	    }
        }
    }

  template<typename Model>
    double
    GillespieSim<Model>::beta_max() const
    {
      if (!pressureCDF_.empty()) {
          typename PressureCDF::const_iterator it = pressureCDF_.end();
          --it;
          return it->first;
      }
      else
        return 0.0;
    }

  template<typename Model>
    void
    GillespieSim<Model>::infect(Infection* infection)
    {
      numS_--;
      numI_++;

      // Choose notification and removal times
      double N = infection->getTime() + model_.ItoN(random_);
      double R = N + model_.NtoR();

      // Set times
      Events events;
      events.I = infection->getTime();
      events.N = N;
      events.R = R;
      if(!population_.updateEvents(*(infection->getIndividual()), events))
        throw runtime_error("Event time modification failed!");

      // Add infectious pressure to individuals preceeding in the CDF
      typename PressureCDF::iterator infectee =
          infection->getPressureIterator();

      double myPressure;
      if (infectee == pressureCDF_.begin())
        myPressure = infectee->first;
      else {
          typename PressureCDF::iterator tmp = infectee;
          --tmp;
          myPressure = infectee->first - tmp->first;
      }

      double cumPressure = 0.0;

      typename PressureCDF::iterator it;
      for (it = pressureCDF_.begin(); it != infectee; it++)
        {
          cumPressure += model_.beta(*(infection->getIndividual()),
              *(it->second),infection->getTime());
          const_cast<double&> (it->first) += cumPressure;
        }

      it = infectee;
      it++;
      for (/*it = it*/; it != pressureCDF_.end(); it++)
        {
          cumPressure += model_.beta(*(infection->getIndividual()),
              *(it->second), infection->getTime());
          const_cast<double&> (it->first) += cumPressure - myPressure;
        }

      pressureCDF_.erase(infectee);

      // Add notification and removal events to event queue
      pair<typename EventQueue::iterator, bool> rv;
      rv = eventQueue_.insert(shared_ptr<Notification> (new Notification(
          infection->getIndividual(), events.N)));
      if (rv.second != true) throw logic_error("Cannot insert notification event!");

      rv = eventQueue_.insert(shared_ptr<Removal> (new Removal(
          infection->getIndividual(), events.R)));
      if(rv.second != true) throw logic_error("Cannot insert removal event!");

      numInfecs_++;
    }

  template<typename Model>
    void
    GillespieSim<Model>::notify(Notification* notification)
    {
      numI_--;
      numN_++;
      // Pop the event off the event queue
      eventQueue_.erase(eventQueue_.begin());

      // Update the pressureCDF
      double cumPressure = 0.0;
      for (typename PressureCDF::iterator it = pressureCDF_.begin(); it
          != pressureCDF_.end(); it++)
        {
          cumPressure += model_.betastar(*(notification->getIndividual()),
              *(it->second), notification->getTime());
          cumPressure -= model_.beta(*(notification->getIndividual()),
              *(it->second), notification->getTime());
          const_cast<double&> (it->first) += cumPressure;
        }

    }

  template<typename Model>
    void
    GillespieSim<Model>::remove(Removal* removal)
    {
      numN_--;
      numR_++;
      // Pop the event off the event queue
      eventQueue_.erase(eventQueue_.begin());

      // Update the pressure CDF
      double cumPressure = 0.0;
      for (typename PressureCDF::iterator it = pressureCDF_.begin(); it
          != pressureCDF_.end(); it++)
        {
          cumPressure -= model_.betastar(*(removal->getIndividual()),
              *(it->second), removal->getTime());
          const_cast<double&> (it->first) += cumPressure;
        }

    }

  template<typename Model>
    void
    GillespieSim<Model>::simulate(bool simCensoredEvents)
    {
      numS_ = population_.size() - 1;
      numI_ = 1;
      numN_ = 0;
      numR_ = 0;
      // Copy population
      double currentTime = population_.getObsTime();

      // Create the PressureCDF
      if (simCensoredEvents) simulateCensoredEvents();
      initEventQueue();
      calcPressureCDF();

      cerr << "Beta max: " << beta_max() << endl;
      // Simulate forward until maxTime
      while (currentTime <= maxTime_ && !eventQueue_.empty())
        {

          // Get next event
          shared_ptr<Event> event;

#ifndef NDEBUG
          // Time of next infection
          try {
              checkPressureCDF();
          }
          catch (exception& e)
            {
              dumpPressureCDF();
              throw (logic_error&)e;
            }
#endif

          double nextInfecTime = POSINF;
          if (!pressureCDF_.empty())
            nextInfecTime = currentTime + random_.gamma(1, beta_max());

          // Choose next event time
          if ((*eventQueue_.begin())->getTime() < nextInfecTime)
            {
              event = *(eventQueue_.begin());
            }
          else
            {
              // Choose who gets infected
              typename PressureCDF::iterator j = pressureCDF_.upper_bound(
                  random_.uniform(0.0, beta_max()));

              // Retrospective sample for either true infection or ghost
              if (isInfectious(j->second, nextInfecTime))
            	  event = shared_ptr<Infection> (new Infection(j, nextInfecTime));
              else
            	  event = shared_ptr<Ghost> (new Ghost(j->second, nextInfecTime));
            }

          currentTime = event->getTime();

#ifndef NDEBUG
          cerr << "Time " << currentTime << endl;
#endif

          // 2. Update populations
          if (Infection* infection = dynamic_cast<Infection*> (event.get()))
            {
#ifndef NDEBUG
              cerr << "Infecting " << infection->getIndividual()->getId() << " at " << currentTime << endl;
#endif
              infect(infection);

            }
          else if (Ghost* ghost = dynamic_cast<Ghost*> (event.get()))
          {
        	  // Do nothing here
          }
          else if (Notification* notification = dynamic_cast<Notification*> (event.get()))
            {
#ifndef NDEBUG
              cerr << "Notifying " << notification->getIndividual()->getId() << " at " << currentTime << endl;
#endif
              notify(notification);

            }
          else if (Removal* removal = dynamic_cast<Removal*> (event.get()))
            {
#ifndef NDEBUG
              cerr << "Removing " << removal->getIndividual()->getId() << " at " << currentTime << endl;
#endif
              remove(removal);
            }
          else
            {
              string msg("Unidentified event: ");
              msg += typeid(event.get()).name();
              throw logic_error(msg);
            }

          population_.setObsTime(currentTime);

        }

    }

  template<typename Model>
    void
    GillespieSim<Model>::dumpPressureCDF() const
    {
      cerr << "================= Pressure Dump ==================" << endl;
      for (typename PressureCDF::const_iterator it = pressureCDF_.begin(); it
          != pressureCDF_.end(); it++)
        {
          cerr << it->first << "\t" << it->second->getId() << endl;
        }
      cerr << "==================================================" << endl;
      population_.dumpInfected();
      cerr << "==================================================" << endl;
    }

  template<typename Model>
  void
  GillespieSim<Model>::checkPressureCDF() const
  {
    double prevval;
    typename PressureCDF::const_iterator it = pressureCDF_.begin();
    if (it->first <= 0.0) throw logic_error("First pressure <= 0.0");
    prevval = it->first;
    it++;
    for (int i=0; it != pressureCDF_.end(); it++, i++)
            {
              if (it->first <= prevval) {
                  std::stringstream msg;
                  msg << "Pressure[i+1] <= Pressure[i] : ";
                  msg << "i = " << i << ",  index = " << it->second->getId();
                  throw logic_error(msg.str().c_str());
              }
              prevval = it->first;
            }
  }

  template<typename Model>
    void
    GillespieSim<Model>::serialize(const std::string filename,
        const bool censored) const
    {
      ofstream file(filename.c_str());
      if (!file.is_open())
        throw runtime_error(
            "Cannot open file to serialize epidemic output.  Check your file path.");

      // HEADER
      file << "id,I,N,R,type\n";

      // CONTENT
      for (typename Model::PopulationType::InfectiveIterator it =
          population_.infecBegin(); it != population_.infecEnd(); it++)
        {
          file << it->getId() << "," << it->getI() << "," << it->getN() << ","
              << it->getR() << "," << (it->isDC() ? "DC" : "IP") << "\n";
        }

      file.close();
    }

}

#endif /* GILLESPIESIM_HPP_ */
