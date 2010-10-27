/*
 * Parameter.hpp
 *
 *  Created on: Oct 20, 2010
 *      Author: stsiab
 */

#ifndef PARAMETER_HPP_
#define PARAMETER_HPP_

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

    /*! Returns the value of the prior pdf
     * evaluated at x
     * @param x The value of the prior parameter
     * @return The value of \f$f(x)$\f
     */
    virtual
    double
    operator()(const double& x) {};

  };

  class Parameter
  {

    double value_;
    Prior prior_;

  public:
    Parameter(const double value,Prior prior) : value_(value), prior_(prior) {}
    virtual
    ~Parameter() {};
    Prior& prior()
    {
      return prior_;
    }

    double operator()()
    {
      return value_;
    }
  };

}

#endif /* PARAMETER_HPP_ */
