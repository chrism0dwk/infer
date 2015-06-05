/*
 * PosteriorHDF5Writer.hpp
 *
 *  Created on: Aug 2, 2012
 *      Author: stsiab
 */

#ifndef POSTERIORHDF5WRITER_HPP_
#define POSTERIORHDF5WRITER_HPP_

#include <string>
#include <H5Cpp.h>
#include <H5PacketTable.h>

#include "Likelihood.hpp"
#include "PosteriorWriter.hpp"

#define PARAMCHUNK 1024
#define INFECCHUNK 2048

namespace EpiRisk
{

  class PosteriorHDF5Writer : public PosteriorWriter
  {
  public:
    PosteriorHDF5Writer(std::string filename, Likelihood& likelihood);
    virtual
    ~PosteriorHDF5Writer();
    void
    write();
    void
    flush();

  private:
    H5::H5File* file_;
    FL_PacketTable* paramTable_;
    FL_PacketTable* infecTable_;
    bool isFirstWrite_;

  };

} /* namespace EpiRisk */
#endif /* POSTERIORHDF5WRITER_HPP_ */
