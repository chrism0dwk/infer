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
    double
    operator()(const double x) = 0;

  };

  class UniformPrior : public Prior
  {
    double
    operator()(const double x = 0)
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
    double value_;
    Prior* prior_;

  public:
    Parameter() : value_(0.0), prior_(new UniformPrior), tag_("") {}
    Parameter(const double value,const Prior& prior,const std::string tag) : value_(value),tag_(tag)
    {
      prior_ = prior.clone();
    }
    Parameter(const Parameter& param)
    {
      tag_ = param.tag_;
      value_ = param.value_;
      prior_ = param.prior_->clone();
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
    operator=(const double x)
    {
      value_ = x;
      return *this;
    }
    virtual
    ~Parameter() {
      delete prior_;
    };
    double prior() const
    {
      return (*prior_)(value_);
    }
    operator double () const
    {
      return value_;
    }
    Parameter&
    operator+=(double x)
    {
      value_ += x;
      return *this;
    }
    Parameter&
    operator*=(double x)
    {
      value_ *= x;
      return *this;
    }
    Parameter&
    operator/=(double x)
    {
      value_ /= x;
      return *this;
    }
    Parameter&
    operator-=(double x)
    {
      value_ -= x;
      return *this;
    }
    const std::string&
    getTag() const
    {
      return tag_;
    }

  };

  typedef boost::numeric::ublas::vector<Parameter> Parameters;
  typedef boost::numeric::ublas::slice ParameterSlice;
  typedef std::vector< Parameter* > ParameterView;

}

#endif /* PARAMETER_HPP_ */
