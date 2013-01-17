/*************************************************************************
 *  ./src/Framework/Individual.hpp
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


#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include <set>
#include <vector>
#include <iostream>
#include <math.h>
#include <list>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <gsl/gsl_math.h>


#include "types.hpp"
#include "EpiRiskException.hpp"
#include "Contact.hpp"

#define FP_TOL 0.000

using namespace std;

namespace EpiRisk
{

  /////////////////////////////////
  // ENUMS
  /////////////////////////////////

  enum InfecStatus_e
  {
    SUSCEPTIBLE = 0, INFECTED, NOTIFIED, REMOVED
  };

  struct Events
  {
    double I;
    double N;
    double R;
    std::string type;
  };

  template<class Covars>
    class Individual
    {
    private:
//    struct CmpConnections
//    {
//      bool operator()(const Individual* lhs, const Individual* rhs) const
//      {
//        return lhs->getI() < rhs->getI();
//      }
//    };

    public:

    typedef Covars CovarsType;
    typedef Contact< Individual<Covars> > ContactType;
    typedef set<ContactType> ContactList;
    typedef std::vector<size_t> ConnectionList;


    private:
      string id_;
      Events events_;
      CovarsType covariates_;  // Maybe implement an auto_ptr here to allow for shallow copy??
                               // NB Tried it with a normal pointer, but ended up having serious
                               // problems when constructing temporaries reading data in!!
      bool known_;
      bool isDC_;
      ContactList contacts_;
      ConnectionList connections_;
      eventTime_t contactStart_;
      bool hasBeenInfected_;
      bool infecByContact_;
      bool nonCentred_;


    public:

      Individual(string id, eventTime_t I, eventTime_t N, eventTime_t R, Covars covars=NULL,
          bool isKnown = 0, InfecStatus_e status = SUSCEPTIBLE) : contactStart_(N),
            known_(isKnown), isDC_(false), hasBeenInfected_(false), infecByContact_(false), nonCentred_(false)
      {
        id_ = id;
        events_.I = I;
        events_.N = N;
        events_.R = R;
      };

      Individual(string id, Covars& covariates) :
        contactStart_(POSINF),known_(false),isDC_(false), hasBeenInfected_(false), infecByContact_(false), nonCentred_(false)
      {
        id_ = id;
        events_.I = POSINF;
        events_.N = POSINF;
        events_.R = POSINF;

        covariates_ = covariates;
      };

      // TODO: Implement copy constructor
      // TODO: Implement assignment method

      virtual
      ~Individual()
      {
      }

      // Virtual constructor / copy
//      virtual Individual *
//      create(size_t myLabel, eventTime_t I, eventTime_t N, eventTime_t R,
//          bool isKnown = 0, InfecStatus_e status = SUSCEPTIBLE) const {};
//
//      virtual Individual *
//      clone() const {};

      // Operators
      bool
      operator<(Individual rhs) const
      {
        return this->getI() < rhs.getI();
      }
      ;

      // Data methods
      void
      setCovariates(Covars& covariates)
      {
        covariates_ = covariates;
      }


      const Covars&
      getCovariates() const
      {
        return covariates_;
      }

      void
      setEvents(Events& events)
      {
        events_ = events;
      }

      const Events&
      getEvents() const
      {
        return events_;
      }

      // Time methods
      string
      getId() const
      {
        return id_;
      }

      double
      getI() const
      {
        return events_.I;
      }
      void
      setI(const eventTime_t I)
      {
        events_.I = I;
      }

      double
      getN() const
      {
        return events_.N;
      }

      void
      setN(const eventTime_t N)
      {
        events_.N = N;
      }

      double
      getR() const
      {
        return events_.R;
      }

      void
      setR(const eventTime_t R)
      {
        events_.R = R;
      }

      bool
      isSAt(const double& time) const
      {
        // Returns true if *this is susceptible
        // at time
        return time < events_.I;
      }
      ;

      bool
      isIAt(const double& time) const
      {
        // Returns true if *this is in the Individual
        // class at time

        return events_.I < time && time <= events_.N;
      }
      ;

      bool
      isNAt(const double& time) const
      {
        // Returns true is *this is Notified
        // at time

        return events_.N < time && time <= events_.R;
      }
      ;

      bool
      isRAt(const double& time) const
      {
        // Returns true if *this is Removed
        // at time
        return events_.R < time;
      }
      ;

      double
      getContactStart() const
      {
        return contactStart_;
      }

      bool
      statusAt(const double& time) const
      {
        // Returns the status of the individual
        if (time < events_.I)
          return SUSCEPTIBLE;
        else if (events_.I <= time && time < events_.N)
          return INFECTED;
        else if (events_.N <= time && time < events_.R)
          return NOTIFIED;
        else
          return REMOVED;
      }

      // Connection graph
      const ConnectionList&
      getConnectionList() const
      {
        return connections_;
      }
      void
      setConnections(ConnectionList& connections)
      {
        connections_ = connections;
      }

      void
      sortConnections()
      {
        sort(connections_.begin(),connections_.end());
      }
      // Miscellaneous methods
      bool
      hasBeenInfected() const
      {
        return hasBeenInfected_;
      }

      bool
      isNonCentred() const;

      // Contact tracing methods
      void
      setContacts(ContactList& contacts, double& ctWindowStart);

      void
      addContact(Individual* conSource, const int conType,
          const eventTime_t conTime)
      {
        // Adds a contact to the contact set
        pair<typename ContactList::iterator, bool> rv;
        rv = contacts_.insert(ContactType(conSource, conType, conTime));
        if (rv.second == false)
          throw data_exception("duplicate contact");
      }
      ;

      void
      delContact(typename ContactList::iterator toDelete)
      {
        // Deletes a contact
        contacts_.erase(toDelete);
      }
      ;

      void
      leftTruncateContacts(const double time)
      {
        // Left truncates contacts
        ContactType tempContact(NULL, 0, time);
        typename ContactList::iterator upperBound = contacts_.upper_bound(tempContact);
        contacts_.erase(contacts_.begin(), upperBound);

      }
      ;

      void
      rightTrucateContacts(const double time)
      {
        // Right truncates contacts
        ContactType tempContact(NULL, 0, time);
        typename ContactList::iterator upperBound = contacts_.upper_bound(tempContact);
        contacts_.erase(upperBound, contacts_.end());
      }
      ;

      bool
      hasContacts() const
      {
        // Returns true if the contact list is not empty

        if (!contacts_.empty())
          return true;
        else
          return false;
      }
      ;

      bool
      infecInCTWindow() const
      {
        // Returns true if Individual occurred during
        // contact tracing window
        // NB: If I>contactStart, it is still implicit that I<N
        if (events_.I > contactStart_)
          {
            //      cout << "In Contact Window!" << endl;
            //cout << "ContactStart: " << contactStart << ", I: " << I << endl;
            return true;
          }
        else
          return false;
      }
      ;

      bool
      inCTWindowAt(const double& time) const
      {
        // Returns true if time is within our CT Window

        if (contactStart_ < time && time <= events_.N)
          return true;
        else
          return false;
      }
      ;

      bool
      hasInfecContacts() const
      {
        // Returns true if one or more potentially
        // infectious contacts exist in the list

        typename ContactList::iterator cIter(contacts_.begin());

        if (!hasContacts())
          return false;

        while (cIter != contacts_.end())
          {
            if (cIter->isInfectious())
              return true;
            cIter++;
          }

        return false;
      }

      std::vector<const ContactType*>
      getInfecContacts() const// TODO: We should watch this one, just in case it impacts on performance!!
      {
        // Gets all infectious contacts

        std::vector<const ContactType*> contactList;
        typename ContactList::iterator cIter(contacts_.begin());

        if (!hasContacts())
          return contactList;

        while (cIter != contacts_.end())
          {
            if (cIter->isInfectious())
              {
                contactList.push_back(&(*cIter));
              }
            cIter++;
          }

        return contactList;
      }
      ;

      size_t
      numNonInfecBy(int contactType) const
      {
        // Returns the number of contacts of type contactType
        // that could potentially be infectious (ie originating
        // from a individual in I) before the Individual time.

        int numContact(0);
        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {

            if ((cIter->time - events_.I) <= FP_TOL)
              break; // Don't go beyond I

            if (cIter->type == contactType && cIter->isInfectious())
              numContact++;

            cIter++;
          }
        return numContact;
      }
      ;

      bool
      isInfecContactAt(const double& t) const
      {
        // Returns true if an infectious contact exists at time t
        typename ContactList::iterator cIter(contacts_.begin());

        while (cIter != contacts_.end())
          {
            if (cIter->time - t == 0 && cIter->isInfectious())
              return true;
            else
              cIter++;
          }

        return false;
      }
      ;

      bool
      isInfecByWhoAt(const int contactType, const double& t,
          Individual*& myIndividual) const
      {
        // Returns true if the Individual time coincides
        // with a contact time of type contactType

        typename ContactList::iterator cIter(contacts_.begin());
        bool rv(false);

        while (cIter != contacts_.end())
          {
            if (fabs(cIter->time - t) <= FP_TOL && cIter->type == contactType
                && cIter->isInfectious())
              {
                myIndividual = cIter->source;

                rv = true;
                break;
              }
            cIter++;
          }

        return rv;
      }
      ;
      bool
      isInfecByWho(const int contactType, const Individual* contactSource) const
      {
        // Returns true if self->I is equal to an infectious contact
        // of type contactType, storing the source contact
        // in contactSource

        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == events_.I)
              {
                if (cIter->type == contactType && cIter->isInfectious())
                  {
                    contactSource = cIter->source;
                    return true;
                  }
                else
                  return false;
              }
            cIter++;
          }

        return false;
      }
      ;

      void
      updateInfecByContact()
      {
        // Sets bool Individual::infecByContact to true
        // if I coincides with an infectious contact
        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == events_.I && cIter->isInfectious())
              {
                infecByContact_ = true;
                return;
              }
            cIter++;
          }

        infecByContact_ = false;
      }
      ;

      bool
      isInfecByContact() const
      {
        // Returns true if the Individual time coincides
        // with an infectious contact

        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == events_.I && cIter->isInfectious())
              return true;
            cIter++;
          }

        return false;
      }
      ;
      bool
      isInfecByContact(Individual*& contactSource) const
      {
        // Returns true if self->I is equal to an infectious contact
        // of any type, stores a pointer to the source in contactSource

        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == events_.I && cIter->isInfectious())
              {
                contactSource = cIter->source;
                return true;
              }
            else if (cIter->getTime() == events_.I && !cIter->isInfectious())
              {
                cout << "WARNING in " << __PRETTY_FUNCTION__
                    << ": non-infec contact at Individual time" << endl;
              }
            cIter++;
          }

        return false;
      }
      ;

      bool
      isContactAt(const double time) const
      {
        // Returns true if a contact exists at conTime

        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == time)
              return true;
          }
        return false;
      }
      ;

      bool
      isContactAt(const double time, int type)
      {
        // Returns true is a contact exists at conTime
        // AND puts the contact type into CON_e& type

        typename ContactList::iterator cIter = contacts_.begin();

        while (cIter != contacts_.end())
          {
            if (cIter->time == time)
              {
                type = cIter->type;
                return true;
              }
          }

        return false;
      }
      ;

      size_t
      numContactsByUntil(const int contactType, const double t) const
      {
        // Returns the number of contacts up to just before t

        size_t numContact = 0;
        typename ContactList::iterator cIter = contacts_.begin();

        if (!hasContacts())
          return 0;

        while (cIter != contacts_.end() && (t - cIter->time) > FP_TOL)
          {
            if (cIter->type == contactType && cIter->isInfectious())
              {
                numContact++;
              }
            cIter++;
          }

        return numContact;
      }

      std::vector<const ContactType*>
      getContactsByUntil(const int contactType, const double t) const
      {
        // Returns a vector of pointers to individuals who
        // contacted *this by method contactType up to just
        // before time t.

        std::vector<const ContactType*> contactList;
        typename ContactList::iterator cIter = contacts_.begin();

        // Return empty vector if no contacts exist (time saver)
        if (!hasContacts())
          return contactList;

        // Iterate over contacts and pick out the relevant entries
        while (cIter != contacts_.end() && (t - cIter->time) > FP_TOL)
          {
            if (cIter->type == contactType && cIter->isInfectious())
              {
                contactList.push_back(&(*cIter));
              }
            cIter++;
          }

        return contactList;
      }
      ;

      void
      switchInfecMethod()
      {
        // Switches bool infecByContact

        if (infecByContact_ == false)
          infecByContact_ = true;
        else
          infecByContact_ = true;
      }
      ;

      void
      attachContacts(ContactList& contacts, double contactStart)
      {
        // Attaches a list of contacts to the Individual
        // and then sorts it.

        contactStart_ = contactStart;
        contacts_ = contacts;
      }

    };

}

#endif
