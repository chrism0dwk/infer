/*************************************************************************
 *  ./src/Framework/Contact.hpp
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
/* ./src/Framework/Contact.hpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */

#ifndef CONTACT_HPP
#define CONTACT_HPP

#include "types.hpp"
#include "Individual.hpp"

using namespace std;

namespace EpiRisk {

  template <typename T>
  class Contact
  {

  private:
    T* source_;
    const int type_;
    const eventTime_t time_;

  public:

    Contact(const T* const source, const int type,
            const eventTime_t time) : source_(source), type_(type), time_(time) {};

    Contact(const Contact<T>& c) : source_(c.getSource()), type_(c.getType()), time_(c.getTime()) {};

    Contact
    operator=(const Contact<T>& c)
    {
      source_ = c.getSource();
      const_cast<int&> (type_) = c.type;
      const_cast<eventTime_t&> (time_) = c.time;
      return *this;
    };

    bool
    operator<(const Contact<T>& rhs) const
    {
      return time < rhs.time;
    };

    bool
    operator==(const Contact<T>& rhs) const
    {
      return time_ == rhs.getTime();
    };

    bool
    isInfectious() const
    {
      // Returns true if the contact
      // is infectious at the time of the contact

      if (source_->isIAt(time))
        return true;
      else
        return false;
    };

    T*
    getSource() const
    {
      return source_;
    }

    int
    getType() const
    {
      return type_;
    }

    eventTime_t
    getTime() const
    {
      return time_;
    }


  };

};

#endif
