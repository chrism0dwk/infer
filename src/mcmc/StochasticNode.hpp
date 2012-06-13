/* ./src/mcmc/StochasticNode.hpp
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
/*
 * StochasticNode.hpp
 *
 *  Created on: Apr 11, 2011
 *      Author: stsiab
 */

#ifndef STOCHASTICNODE_HPP_
#define STOCHASTICNODE_HPP_

#include "Parameter.hpp"

namespace EpiRisk
{

  // Stochastic node class
  class StochasticNode
  {
  protected:
    Parameter* param_;
  public:
    StochasticNode(Parameter& param) : param_(&param) {};
    virtual StochasticNode*
    clone()
    {
      return new StochasticNode(*this);
    }
    virtual double getValue() const
    {
      return (double)(*param_);
    }
    virtual void setValue(const double value) {
      (*param_) = value;
    }
    virtual double prior() const{
      return param_->prior();
    }
  };

  //! An UpdateBlock is a collection of StochasticNodes
  //! that are drawn together.
  class UpdateBlock
  {
    std::vector< StochasticNode* > updateblock_;
  public:
    virtual
    ~UpdateBlock()
    {
      for(std::vector< StochasticNode* >::iterator it = updateblock_.begin();
          it != updateblock_.end();
          it++) delete *it;
    }
    void add(Parameter& param)
    {
      StochasticNode* tmp = new StochasticNode(param);
      updateblock_.push_back(tmp);
    }
    void add(StochasticNode* node)
    {
      StochasticNode* tmp = node->clone();
      updateblock_.push_back(tmp);
    }
    StochasticNode*
    operator[](const int idx) const
    {
#ifndef NDEBUG
      if(idx >= updateblock_.size()) throw std::range_error("idx out of range in UpdateBlock");
#endif

      return updateblock_[idx];
    }
    size_t
    size() const
    {
      return updateblock_.size();
    }
  };

}

#endif /* STOCHASTICNODE_HPP_ */
