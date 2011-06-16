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

#include "Random.hpp"
#include "Model.hpp"

#define POSINF (numeric_limits<double>::infinity())

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
          return time_ < rhs.time_;
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
        virtual
        const typename Model::Individual*
        getIndividual() const
        {
          return this->individual_;
        }
        virtual
        typename PressureCDF::iterator
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
        ;
      };

      class Removal : public Event
      {
      public:
        Removal(const typename Model::Individual* individual, double time) :
          Event(individual, time)
        {
        }
        ;
      };

      typedef set<shared_ptr<Event> > EventQueue;

      Model& model_;
      Random& random_;
      double maxTime_;
      double currentTime_;
      PressureCDF pressureCDF_;
      EventQueue eventQueue_;
      typename Model::PopulationType& population_;

      void
      calcPressureCDF();
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
      model_(model), random_(random), population_(model_.getPopulation()),maxTime_(POSINF)
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
    typename Model::PopulationType::InfectiveIterator stop = population_.infecLowerBound(currentTime_);
    for(typename Model::PopulationType::InfectiveIterator it = population_.infecBegin();
        it != stop;
        it++)
      {
        if (it->getN() > currentTime_ && it->getN() < POSINF)
          {
            eventQueue_.insert(shared_ptr<Notification>(new Notification(&(*it),it->getN())));
          }

        if (it->getR() > currentTime_ && it->getR() < POSINF)
          {
            eventQueue_.insert(shared_ptr<Removal>(new Removal(&(*it),it->getR())));
          }
      }
  }

  template<typename Model>
    double
    GillespieSim<Model>::beta_max() const
    {
    if(!pressureCDF_.empty()) return (--pressureCDF_.end())->first;
    else return 0.0;
    }


template<typename Model>
  void
  GillespieSim<Model>::infect(Infection* infection)
  {
    // Choose notification and removal times
    double N = infection->getTime() + model_.ItoN(random_);
    double R = N + model_.NtoR();

    // Set times
    Events events;
    events.I = infection->getTime();
    events.N = N;
    events.R = R;
    population_.updateEvents(*(infection->getIndividual()),events);

    // Add infectious pressure to individuals preceeding in the CDF
    typename PressureCDF::iterator infectee = infection->getPressureIterator();

    double myPressure;
    if(infectee == pressureCDF_.begin()) myPressure = infectee->first;
    else myPressure = infectee->first - (--infectee)->first;

    double cumPressure = 0.0;

    typename PressureCDF::iterator it;
    for(it = pressureCDF_.begin();
        it != infectee;
        it++)
      {
        cumPressure += model_.beta(*(infection->getIndividual()), *(it->second));
        const_cast<double&> (it->first) += cumPressure;
      }

    it = infectee;
    it++;
    for(it = it;
            it != pressureCDF_.end();
            it++)
      {
        cumPressure += model_.beta(*(infection->getIndividual()), *(it->second));
        const_cast<double&> (it->first) += cumPressure - myPressure;
      }

    pressureCDF_.erase(infectee);

    // Add notification and removal events to event queue
    eventQueue_.insert(shared_ptr<Notification>(new Notification(infection->getIndividual(),N)));
    eventQueue_.insert(shared_ptr<Removal>(new Removal(infection->getIndividual(),R)));
  }

template<typename Model>
  void
  GillespieSim<Model>::notify(Notification* notification)
  {
    // Pop the event off the event queue
    eventQueue_.erase(eventQueue_.begin());

    // Update the pressureCDF
    double cumPressure = 0.0;
    for(typename PressureCDF::iterator it = pressureCDF_.begin();
        it != pressureCDF_.end();
        it++)
      {
        cumPressure += model_.betastar(*(notification->getIndividual()),*(it->second));
        cumPressure -= model_.beta(*(notification->getIndividual()), *(it->second));
        const_cast<double&> (it->first) += cumPressure;
      }

  }

template<typename Model>
  void
  GillespieSim<Model>::remove(Removal* removal)
  {
    // Pop the event off the event queue
  eventQueue_.erase(eventQueue_.begin());

  // Update the pressure CDF
  double cumPressure = 0.0;
  for(typename PressureCDF::iterator it = pressureCDF_.begin();
      it != pressureCDF_.end();
      it++)
    {
      cumPressure -= model_.betastar(*(removal->getIndividual()),*(it->second));
      const_cast<double&> (it->first) += cumPressure;
    }

  }

template<typename Model>
  void
  GillespieSim<Model>::simulate()
  {

    // Copy population
    currentTime_ = model_.getPopulation().getObsTime();

    // Create the PressureCDF
    calcPressureCDF();
    initEventQueue();

    // Simulate forward until maxTime
    while (currentTime_ <= maxTime_ && !eventQueue_.empty())
      {
        cout << "loop" << endl;
        // Get next event
        shared_ptr < Event > event;

        // Time of next infection
        if(beta_max() < 0) {
            dumpPressureCDF();
            throw logic_error("Weirdness in CDF!");
        }

        double nextInfecTime = POSINF;
        if(!pressureCDF_.empty()) nextInfecTime = currentTime_ + random_.gamma(1, beta_max());
        cout << "Next infec event time: " << nextInfecTime << endl;
        // Choose next event
        if (eventQueue_.begin()->get()->getTime() < nextInfecTime)
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

        // 2. Update populations
        if (Infection* infection = dynamic_cast<Infection*> (event.get()))
          {
            cout << "Infection" << endl;
            infect(infection);
          }
        else if (Notification* notification = dynamic_cast<Notification*> (event.get()))
          {
            cout << "Notification" << endl;
            notify(notification);
          }
        else if (Removal* removal = dynamic_cast<Removal*> (event.get()))
          {
            cout << "Removal" << endl;
            remove(removal);
          }
        else
          {
            string msg("Unidentified event: ");
            msg += typeid(event.get()).name();
            throw logic_error(msg);
          }

        currentTime_ = event.get()->getTime();

      }

  }

template<typename Model>
void
GillespieSim<Model>::dumpPressureCDF() const
{
  cout << "================= Pressure Dump ==================" << endl;
  for(typename PressureCDF::const_iterator it = pressureCDF_.begin();
      it != pressureCDF_.end();
      it++)
    {
      cout << it->first << "\t" << it->second->getId() << endl;
    }
  cout << "==================================================" << endl;
}

template<typename Model>
  void
  GillespieSim<Model>::serialize(const std::string filename,
      const bool censored) const
  {
    ofstream file(filename.c_str());
    if (!file.is_open()) throw runtime_error("Cannot open file to serialize epidemic output.  Check your file path.");

    // HEADER
    file << "id,I,N,R\n";

    // CONTENT
    for(typename Model::PopulationType::InfectiveIterator it = population_.infecBegin();
        it != population_.infecEnd();
        it++)
      {
        file << it->getId() << "," << it->getI() << "," << it->getN() << "," << it->getR() << "\n";
      }

    file.close();
  }

}

#endif /* GILLESPIESIM_HPP_ */
