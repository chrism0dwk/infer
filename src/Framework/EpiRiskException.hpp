/*************************************************************************
 *  ./src/Framework/EpiRiskException.hpp
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
 * EpiRiskException.hpp
 *
 *  Created on: 22 Jan 2010
 *      Author: stsiab
 */

#ifndef EPIRISKEXCEPTION_HPP_
#define EPIRISKEXCEPTION_HPP_

#include <exception>
#include <cstdio>

namespace EpiRisk {

  class data_exception : public std::exception
  {
  public:
    data_exception(const char* msg) {
      msg_ = msg;
    }
    virtual const char* what() const throw()
    {
      return msg_;
    }

  private:
    const char* msg_;
  };


  class output_exception : public std::exception
  {
  public:
    output_exception(const char* msg) {
      msg_ = msg;
    }
    virtual const char* what() const throw()
    {
      return msg_;
    }

  private:
    const char* msg_;
  };


  class parse_exception : public std::exception
   {
   public:
     parse_exception(const char* msg) {
       msg_ = msg;
     }
     virtual const char* what() const throw()
     {
       return msg_;
     }

   private:
     const char* msg_;
   };

  class range_exception : public std::exception
     {
     public:
       range_exception(const char* msg) {
         sprintf(msg_,"Range exception: %s",msg);
       }
       virtual const char* what() const throw()
       {
         return msg_;
       }

     private:
       char msg_[200];
     };

  class param_exception : public std::exception
  {
  public:
	  param_exception(const char* msg)
	  {
		  msg_ = msg;
	  }
	  virtual const char* what() const throw()
		{
		  return msg_;
		}

  private:
	  const char* msg_;
  };

  class cholesky_error : public std::exception
  {
  public:
          cholesky_error(const char* msg)
          {
                  msg_ = msg;
          }
          virtual const char* what() const throw()
                {
                  return msg_;
                }

  private:
          const char* msg_;
  };

  class fileEOF : public std::exception
  {
  public:
    fileEOF() : exception() {};
    virtual const char* what() const throw()
    {
      return "End of file";
    }
  };




}


#endif /* EPIRISKEXCEPTION_HPP_ */
