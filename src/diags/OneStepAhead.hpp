/*************************************************************************
 *  ./src/diags/OneStepAhead.hpp
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
 * onestepahead.cpp
 *
 *  Created on: 29 Sep 2011
 *      Author: stsiab
 */

#include <set>
#include <string>
#include <gsl/gsl_cdf.h>

#include "types.hpp"
#include "Random.hpp"
#include "Model.hpp"

namespace EpiRisk
{

  template<typename ModelT>
    class OneStepAhead
    {
    public:
      OneStepAhead(const ModelT& model, const string outputfilename) :
        model_(model)
      {
        outputFile_.open(outputfilename.c_str(),ios::out);
        if(!outputFile_.is_open())
          {
            string msg = "Cannot open output file '";
            msg += outputfilename;
            msg += "'";
            throw runtime_error(msg.c_str());
          }
      }
      ~OneStepAhead()
      {
        outputFile_.close();
      }
      void
      compute()
      {
        // Construct the event list
        events_.clear();
        for(typename ModelT::PopulationType::InfectiveIterator it = (++model_.getPopulation().infecBegin());
            it != model_.getPopulation().infecEnd();
            it++)
          {
            events_.insert(Event(Event::INFECTION,it->getI(),it));
            events_.insert(Event(Event::NOTIFICATION,it->getN(),it));
            events_.insert(Event(Event::REMOVAL,it->getR(),it));
          }
        std::cout << "Events size = " << events_.size() << std::endl;

        // Find infection events, and calculate the one step ahead F(x)
        bool isFirst = true;
        for(typename std::set<Event>::const_iterator it = (++events_.begin());
            it != events_.end();
            it++)
          {

            if(it->getType() == Event::INFECTION)
              {
                typename std::set<Event>::const_iterator prevEvent = it;
                prevEvent--;
                double timediff = it->getTime() - prevEvent->getTime();
                double pressure = model_.instantPressureOn(it->getIndividual(), it->getTime());
                if(!isFirst) {
                    outputFile_ << " ";
                }
                else isFirst = false;
                outputFile_ << /*it->getIndividual()->getId() << ":" <<*/ gsl_cdf_exponential_P(timediff,1.0/pressure);
              }

          }
        outputFile_ << "\n";
      }

    private:
      const ModelT& model_;
      ofstream outputFile_;
      class Event
      {
      public:

        enum Type
        {
          INFECTION,NOTIFICATION,REMOVAL
        };

        Event(const Type type, const double time, const typename ModelT::PopulationType::InfectiveIterator& individual) :
          type_(type),individual_(individual),time_(time)
        {
        }
        virtual ~Event() {}

        bool operator<(const Event& rhs) const
        {
          return time_ < rhs.time_;
        }

        const typename ModelT::PopulationType::InfectiveIterator& getIndividual() const
        {
          return individual_;
        }

        Type getType() const
        {
          return type_;
        }

        double getTime() const
        {
          return time_;
        }
      private:
        Type type_;
        const typename ModelT::PopulationType::InfectiveIterator individual_;
        double time_;
      };

      std::multiset<Event> events_;

    };

}

