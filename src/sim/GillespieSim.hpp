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
      simulate();
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
      EventQueue eventQueue_;
      typename Model::PopulationType& population_;
      int numS_, numI_, numN_, numR_;
      size_t numInfecs_;

      void
      calcPressureCDF();
      void
      checkPressureCDF() const;
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

      for (typename Model::PopulationType::PopulationIterator it =
          population_.asPop(population_.infecEnd()); it != population_.end(); it++)
        {
          cumulativePressure += model_.instantPressureOn(population_.asI(it),
              model_.getObsTime());
          pressureCDF_.insert(pair<double, const typename Model::Individual*> (
              cumulativePressure, &(*it)));
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
              eventQueue_.insert(shared_ptr<Notification> (new Notification(
                  &(*it), it->getN())));
            }

          if (it->getR() > population_.getObsTime() && it->getR() < POSINF)
            {
              eventQueue_.insert(shared_ptr<Removal> (new Removal(&(*it),
                  it->getR())));
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
              *(it->second));
          const_cast<double&> (it->first) += cumPressure;
        }

      it = infectee;
      it++;
      for (it = it; it != pressureCDF_.end(); it++)
        {
          cumPressure += model_.beta(*(infection->getIndividual()),
              *(it->second));
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
              *(it->second));
          cumPressure -= model_.beta(*(notification->getIndividual()),
              *(it->second));
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
              *(it->second));
          const_cast<double&> (it->first) += cumPressure;
        }

    }

  template<typename Model>
    void
    GillespieSim<Model>::simulate()
    {
      numS_ = population_.size() - 1;
      numI_ = 1;
      numN_ = 0;
      numR_ = 0;
      // Copy population
      double currentTime = population_.getObsTime();

      // Create the PressureCDF
      calcPressureCDF();
      initEventQueue();

      // Simulate forward until maxTime
      while (currentTime <= maxTime_ && !eventQueue_.empty())
        {

          // Get next event
          shared_ptr<Event> event;

          // Time of next infection
          try {
              checkPressureCDF();
          }
          catch (exception& e)
            {
              dumpPressureCDF();
              throw (logic_error&)e;
            }

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
              // Choose infection
              typename PressureCDF::iterator it = pressureCDF_.upper_bound(
                  random_.uniform(0.0, beta_max()));
              event = shared_ptr<Infection> (new Infection(it, nextInfecTime));
            }

          currentTime = event->getTime();

          // 2. Update populations
          if (Infection* infection = dynamic_cast<Infection*> (event.get()))
            {
              infect(infection);
            }
          else if (Notification* notification = dynamic_cast<Notification*> (event.get()))
            {
              notify(notification);
            }
          else if (Removal* removal = dynamic_cast<Removal*> (event.get()))
            {
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
      cout << "================= Pressure Dump ==================" << endl;
      for (typename PressureCDF::const_iterator it = pressureCDF_.begin(); it
          != pressureCDF_.end(); it++)
        {
          cout << it->first << "\t" << it->second->getId() << endl;
        }
      cout << "==================================================" << endl;
    }

  template<typename Model>
  void
  GillespieSim<Model>::checkPressureCDF() const
  {
    for (typename PressureCDF::const_iterator it = pressureCDF_.begin(); it
              != pressureCDF_.end(); it++)
            {
              if (it->first <= 0.0) throw logic_error("Pressure <= 0.0!");
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
      file << "id,I,N,R\n";

      // CONTENT
      for (typename Model::PopulationType::InfectiveIterator it =
          population_.infecBegin(); it != population_.infecEnd(); it++)
        {
          file << it->getId() << "," << it->getI() << "," << it->getN() << ","
              << it->getR() << "\n";
        }

      file.close();
    }

}

#endif /* GILLESPIESIM_HPP_ */
