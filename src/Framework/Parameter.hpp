/*************************************************************************
 *  ./src/Framework/Parameter.hpp
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
 * Parameter.hpp
 *
 *  Created on: Oct 20, 2010
 *      Author: stsiab
 */

#ifndef PARAMETER_HPP_
#define PARAMETER_HPP_


#include <string>
#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

namespace EpiRisk
{
  /*! \brief Interface to EpiRisk prior
   *
   * The Prior class provides an interface to a prior distribution
   * used by the Parameter class.
   */
  class Prior
  {
  public:
    Prior() {};
    virtual
    ~Prior() {};
    virtual
    Prior* clone() const = 0;
    virtual
    Prior* create() const = 0;

    /*! Returns the value of the prior pdf
     * evaluated at x
     * @param x The value of the prior parameter
     * @return The value of \f$f(x)$\f
     */
    virtual
    float
    operator()(const float x) = 0;

  };

  class UniformPrior : public Prior
  {
    float
    operator()(const float x = 0)
    {
      return 1.0;
    }
    Prior*
    clone() const
    {
      return new UniformPrior(*this);
    }
    Prior*
    create() const
    {
      return new UniformPrior;
    }
  };


  class Parameter
  {
    std::string tag_;
    float value_;
    Prior* prior_;

  public:
    Parameter() : value_(0.0), prior_(new UniformPrior), tag_("") {};
    Parameter(const float value,const Prior& prior,const std::string tag) : value_(value), tag_(tag)
    {
      prior_ = prior.clone();
    }
    Parameter(const Parameter& param)
    {
      tag_ = param.tag_;
      value_ = param.value_;
      prior_ = param.prior_->clone();
    }
    virtual Parameter* clone()
    {
      return new Parameter(*this);
    }
    virtual Parameter* create()
    {
      return new Parameter();
    }
    const std::string&
    GetTag() const
    {
      return tag_;
    }
    Parameter&
    operator=(const Parameter& param)
    {
      if(this != &param) {
          delete prior_;
          tag_ = param.tag_;
          value_ = param.value_;
          prior_ = param.prior_->clone();
      }
      return *this;
    }
    Parameter&
    operator=(const float x)
    {
      value_ = x;
      return *this;
    }
    virtual
    ~Parameter() {
      delete prior_;
    };
    float prior() const
    {
      return (*prior_)(value_);
    }
    operator float () const
    {
      return value_;
    }
    Parameter&
    operator+=(float x)
    {
      value_ += x;
      return *this;
    }
    Parameter&
    operator*=(float x)
    {
      value_ *= x;
      return *this;
    }
    Parameter&
    operator/=(float x)
    {
      value_ /= x;
      return *this;
    }
    Parameter&
    operator-=(float x)
    {
      value_ -= x;
      return *this;
    }
    float*
    GetValuePtr()
    {
      return &value_;
    }
    float
    GetValue() const
    {
      return value_;
    }
  };

  typedef boost::numeric::ublas::vector<Parameter> Parameters;
  typedef boost::numeric::ublas::slice ParameterSlice;

  typedef boost::ptr_vector<Parameter> ParameterSerializerList;

  class ParameterSerializer
  {
  public:
    ParameterSerializer(ParameterSerializerList& params, std::ostream& os) : params_(params)
    {
      ParameterSerializerList::const_iterator it = params_.begin();
      os << it->GetTag();
      ++it;
      while(it != params_.end())
        {
          os << "," << it->GetTag();
          ++it;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const ParameterSerializer& paramSerializer);

  private:
    ParameterSerializerList& params_;
  };

  std::ostream& operator<<(std::ostream& os, const ParameterSerializer& paramSerializer);


}

#endif /* PARAMETER_HPP_ */
