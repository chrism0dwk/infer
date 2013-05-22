/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
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
#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include <set>
#include <vector>
#include <iostream>
#include <math.h>
#include <list>
#include <stdexcept>

#include "SmpTypes.hpp"


using namespace std;

namespace EpiRisk
{
  namespace Smp {
  /////////////////////////////////
  // FWD DECLS
  /////////////////////////////////

  class Contact;
  class Individual;

  /////////////////////////////////
  // ENUMs
  /////////////////////////////////


//  enum CON_e
//  {
//    // Connection types
//    FEEDMILL,
//    SHOUSE,
//    COMPANY
//  };






  class Individual
  {
  public:

    typedef set<Contact> ContactList;
    typedef vector<size_t> ConnectionList;

    enum infecStatus_e
    {
      SUSCEPTIBLE = 0, INFECTED, NOTIFIED, REMOVED
    };

    Ilabel_t label;
    eventTime_t I; // Changes
    eventTime_t U; // Non-centred infection time
    eventTime_t N;
    eventTime_t R;
    bool known;
    bool isDC;
    ContactList contacts;
    ConnectionList connections;
    eventTime_t contactStart;
    double sum_beta;
    double sum_beta_can;
    bool hasBeenInfected;
    bool infecByContact;
    bool nonCentred;

    bool
    operator<(Individual) const;

    Individual(size_t myLabel, eventTime_t myI, eventTime_t myN,
        eventTime_t myR, bool isKnown = 0, infecStatus_e _status = SUSCEPTIBLE); // Constructor for an infection
    virtual
    ~Individual();

    //  // Virtual constructor / copy
    //  virtual Individual * create(size_t myLabel,
    //            eventTime_t myI,
    //            eventTime_t myN,
    //            eventTime_t myR,
    //            bool isKnown=0,
    //            infecStatus_e _status=SUSCEPTIBLE) const = 0;
    //
    //  virtual Individual * clone() const = 0;

    void
    attachContacts(ContactList&, double&);
    void
    addContact(Individual* conSource, const int conType,
        const eventTime_t conTime);
    void
    delContact(ContactList::iterator toDelete);
    void
    leftTruncateContacts(const double time);
    void
    rightTrucateContacts(const double time);
    bool
    hasContacts();
    bool
    isSAt(const double&);
    bool
    isIAt(const double&);
    bool
    isNAt(const double&);
    bool
    isRAt(const double&);
    bool
    statusAt(const double&);
    bool
    infecInCTWindow();
    bool
    inCTWindowAt(const double&);
    bool
    hasInfecContacts();
    vector<const Contact*>
    getInfecContacts();
    size_t
    numNonInfecBy(int);
    bool
    isInfecContactAt(const double&);
    bool
    isInfecByWhoAt(const int, const double&, Individual*&);
    bool
    isInfecByWho(const int, const Individual*);
    void
    updateInfecByContact();
    bool
    isInfecByContact();
    bool
    isInfecByContact(Individual*&);
    bool
    isContactAt(const double);
    bool
    isContactAt(const double, int&);
    size_t
    numContactsByUntil(const int, const double);
    vector<const Contact*>
    getContactsByUntil(const int, const double);
    void
    switchInfecMethod();

  };

  ///////////////////////////////////////////////////
  // Classes related to contact tracing
  ///////////////////////////////////////////////////



  using namespace std;

  class Contact
  {
  public:
    Individual* source;
    const int type;
    const eventTime_t time;

    Contact(Individual* conSource, const int conType,
        const eventTime_t conTime);
    Contact(const Contact&);
    Contact
    operator=(const Contact&);
    bool
    operator<(const Contact&) const;
    bool
    operator==(const Contact&) const;
    bool
    isInfectious() const;
  };

  } }

#endif
