/*************************************************************************
 *  ./src/mcmc/PosteriorWriter.hpp
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
 * PosteriorWriter.hpp
 *
 *  Created on: 22 Jul 2011
 *      Author: stsiab
 */

#ifndef POSTERIORWRITER_HPP_
#define POSTERIORWRITER_HPP_

#include <boost/function.hpp>

#include "Parameter.hpp"

namespace EpiRisk
{

  // FWD decl
  class GpuLikelihood;

  class PosteriorWriter
  {
  public:
    PosteriorWriter(GpuLikelihood& likelihood);
    virtual
    ~PosteriorWriter();
    void
    AddParameter(Parameter& param);
    template<class F>
    void
    AddSpecial(std::string tag, F& functor) {
      paramTags_.push_back(tag);
      special_.push_back(boost::function<float ()>(functor));
    }
    virtual
    void
    write() = 0;

  protected:
    GpuLikelihood& likelihood_;
    std::vector<Parameter*> paramVals_; std::vector<float> valueBuff_;
    std::vector<std::string> paramTags_; std::vector<float> infecBuff_;
    std::vector< boost::function<float ()> > special_;

    void
    GetParamVals(float* buff);
  };

}

#endif /* POSTERIORWRITER_HPP_ */
