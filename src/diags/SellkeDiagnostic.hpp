/*************************************************************************
 *  ./src/diags/SellkeDiagnostic.hpp
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
#include <sstream>
#include <cstdio>
#include <sys/file.h>

#include "types.hpp"
#include "Random.hpp"
#include "Model.hpp"

namespace EpiRisk
{

  template<typename ModelT>
    class SellkeDiagnostic
    {
    public:
      SellkeDiagnostic(const ModelT& model, const string outputfilename) :
        model_(model)
      {
        outputFile_ = fopen(outputfilename.c_str(),"ab");
        if(outputFile_ == NULL)
          {
            string msg = "Cannot open output file '";
            msg += outputfilename;
            msg += "'";
            throw runtime_error(msg.c_str());
          }

        string indexFilename = outputfilename + ".idx";
        indexFile_ = fopen(indexFilename.c_str(),"ab");
        if(indexFile_ == NULL)
          {
            string msg = "Cannot open index file";
            throw runtime_error(msg.c_str());
          }
      }
      ~SellkeDiagnostic()
      {
        if (fileLocked_) unlockfiles();
        fclose(outputFile_);
        fclose(indexFile_);
      }
      void
      compute(const size_t idx)
      {

        stringstream output;

        // Construct the event list
        events_.clear();
        for(typename ModelT::PopulationType::InfectiveIterator it = model_.getPopulation().infecBegin();
            it != model_.getPopulation().infecEnd();
            it++)
          {
            events_.insert(Event(Event::INFECTION,it->getI(),it));
            if(it->getN() != POSINF) events_.insert(Event(Event::NOTIFICATION,it->getN(),it));
            if(it->getR() != POSINF) events_.insert(Event(Event::REMOVAL,it->getR(),it));
          }
        std::cout << "Events size = " << events_.size() << std::endl;

        // Iterate over the event list
        bool firstTimeStep = true;
        for(typename std::set<Event>::const_iterator event = (++events_.begin());
            event != events_.end();
            event++)
          {
            typename std::set<Event>::const_iterator prevEvent = event; prevEvent--;
            if(!firstTimeStep) output << ";";
            else firstTimeStep = false;

            // Iterate over the population
            bool firstIndividual = true;
            for(typename ModelT::PopulationType::PopulationIterator indiv = model_.getPopulation().begin();
                indiv != model_.getPopulation().end();
                indiv++)
                {
                  if(indiv->getI() >= event->getTime()) {
                      double sellke = model_.instantPressureOn(model_.getPopulation().asI(indiv), event->getTime());
                      double timediff = event->getTime() - prevEvent->getTime();
                      if(timediff == 0.0) continue; // Exclude null epochs

                      if(!firstIndividual) output << ",";
                      else firstIndividual = false;
                      output << indiv->getId();
                      if(event->getTime() == indiv->getI()) output << ":1:";
                      else output << ":0:";
                      output << sellke*timediff;
                  }
                }
          }
        output << "\n";

        // Write to output files
         try {
             stringstream s; s << idx << ":";
             lockfiles();
             fseek(outputFile_,0,SEEK_END);
             fseek(indexFile_,0,SEEK_END);
             s << ftell(outputFile_) << "\n";
             fputs(s.str().c_str(),indexFile_);
             fputs(output.str().c_str(),outputFile_);
             fflush(outputFile_);
             fflush(indexFile_);
             unlockfiles();
         }
         catch (runtime_error& e)
         {
             throw e;
         }
      }

    private:
      const ModelT& model_;
      FILE* outputFile_;
      FILE* indexFile_;
      bool fileLocked_;
      void
      lockfiles()
      {
        int rv;
        rv = flock(fileno(outputFile_),LOCK_EX);
        if (rv == -1) throw runtime_error("Cannot lock output file");

        rv = flock(fileno(indexFile_),LOCK_EX);
        if (rv == -1) throw runtime_error("Cannot lock index file");
        fileLocked_ = true;
      }

      void
      unlockfiles()
      {
        flock(fileno(outputFile_),LOCK_UN);
        flock(fileno(indexFile_),LOCK_UN);
        fileLocked_ = false;
      }

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

