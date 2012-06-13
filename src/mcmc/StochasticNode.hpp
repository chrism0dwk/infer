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
