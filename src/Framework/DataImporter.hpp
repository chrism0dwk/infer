/*
 * DataImporter.hpp
 *
 *  Created on: Oct 12, 2010
 *      Author: stsiab
 */

#ifndef DATAIMPORTER_HPP_
#define DATAIMPORTER_HPP_

#include "EpiRiskException.hpp"

namespace EpiRisk
{
  template < class T >
  class DataImporter
  {
  public:
    typedef T DataType;
    struct Record
     {
       string id;
       T data;
     };

    DataImporter() {};

    virtual
    ~DataImporter() {};

    virtual
    void
    open() = 0;

    virtual
    void
    close() = 0;

    virtual
    Record
    next() = 0;

    virtual
    void
    reset() = 0;
  };

}

#endif /* DATAIMPORTER_HPP_ */
