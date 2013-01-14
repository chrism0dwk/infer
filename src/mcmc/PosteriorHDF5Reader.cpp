/*
 * PosteriorHDF5Reader.cpp
 *
 *  Created on: Dec 17, 2012
 *      Author: cpjewell
 */

#include <H5PacketTable.h>
#include <stdexcept>
#include <malloc.h>
#include <iostream>

#include "PosteriorHDF5Reader.hpp"
#include "EpiRiskException.hpp"

namespace
  {
    hsize_t
    getFLPTwidth(H5::DataSet& dataset)
    {
      H5::ArrayType pType = dataset.getArrayType();
      hsize_t* dims = new hsize_t[pType.getArrayNDims()]; // Check is 1!
      pType.getArrayDims(dims); // Stored
      hsize_t numFields = dims[0];
      delete[] dims;

      return numFields;
    }
  }

namespace EpiRisk
{

  PosteriorHDF5Reader::PosteriorHDF5Reader(const std::string filename) : filename_(filename)
  {
    try
      {
        theFile_ = new H5::H5File(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT,
            H5P_DEFAULT);

        char paramPath[] = PARAMPATH;
        parameters_ = theFile_->openDataSet(paramPath);
        parametersPt_ = new FL_PacketTable(theFile_->getId(), paramPath);

        char infecPath[] = INFECPATH;
        infections_ = theFile_->openDataSet(infecPath);
        infectionsPt_ = new FL_PacketTable(theFile_->getId(), infecPath);

        size_ = parametersPt_->GetPacketCount();
        numParams_ = getFLPTwidth(parameters_);
      }
    catch (H5::Exception& e)
      {
        throw parse_exception("Error closing posterior file");
      }
    catch (std::runtime_error& e)
      {
        throw e;
      }
    catch (...)
      {
        throw parse_exception("Unknown exception closing file");
      }

  }


PosteriorHDF5Reader::PosteriorHDF5Reader(const PosteriorHDF5Reader& other) : filename_(other.filename_)
{
  try
        {
          theFile_ = new H5::H5File(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT,
              H5P_DEFAULT);

          char paramPath[] = PARAMPATH;
          parameters_ = theFile_->openDataSet(paramPath);
          parametersPt_ = new FL_PacketTable(theFile_->getId(), paramPath);

          char infecPath[] = INFECPATH;
          infections_ = theFile_->openDataSet(infecPath);
          infectionsPt_ = new FL_PacketTable(theFile_->getId(), infecPath);

          size_ = parametersPt_->GetPacketCount();
          numParams_ = getFLPTwidth(parameters_);
        }
      catch (H5::Exception& e)
        {
          throw parse_exception("Error closing posterior file");
        }
      catch (std::runtime_error& e)
        {
          throw e;
        }
      catch (...)
        {
          throw parse_exception("Unknown exception closing file");
        }
}




  PosteriorHDF5Reader::~PosteriorHDF5Reader()
  {
    try
      {
        delete parametersPt_;
        delete infectionsPt_;
        delete theFile_;
      }
    catch (H5::Exception& e)
      {
        throw parse_exception("Error closing posterior file");
      }
    catch (std::runtime_error& e)
      {
        throw e;
      }
    catch (...)
      {
        throw parse_exception("Unknown exception closing file");
      }
  }

  size_t
  PosteriorHDF5Reader::GetSize() const
  {
    return size_;
  }

  void
  PosteriorHDF5Reader::GetParameterTags(std::vector<std::string>& tags)
  {
    // Retrieves parameter names

    H5::Attribute theTags = parameters_.openAttribute("tags");

    H5::DataSpace tagDS = theTags.getSpace();
    hsize_t* aDim = new hsize_t[tagDS.getSimpleExtentNdims()];
    tagDS.getSimpleExtentDims(aDim);

    char** readBuff = new char*[aDim[0]];
    H5::DataType paramTag_t = theTags.getDataType();

    theTags.read(paramTag_t, readBuff);
    tags.resize(aDim[0]);
    for (int i = 0; i < aDim[0]; ++i)
      {
        tags[i] = std::string(readBuff[i]);
      }

    H5Dvlen_reclaim(paramTag_t.getId(), tagDS.getId(), H5P_DEFAULT, readBuff);
    delete[] readBuff;
    delete[] aDim;
  }

  void
  PosteriorHDF5Reader::GetPopulationIds(std::vector<std::string>& ids)
  {
    // Retrieves population IDs -- for matching to infective index

    char idPath[] = IDPATH;
    H5::DataSet idDataSet = theFile_->openDataSet(idPath);
    H5::DataSpace idDataSpace = idDataSet.getSpace();
    hsize_t* dim = new hsize_t[idDataSpace.getSimpleExtentNdims()];
    idDataSpace.getSimpleExtentDims(dim);

    char** readBuff = new char*[dim[0]];
    H5::DataType idDataType = idDataSet.getDataType();

    idDataSet.read(readBuff, idDataType);

    ids.resize(dim[0]);
    for(int i = 0; i<dim[0]; ++i) {
        ids[i] = std::string(readBuff[i]);
    }

    H5Dvlen_reclaim(idDataType.getId(), idDataSpace.getId(), H5P_DEFAULT, readBuff);
    delete[] readBuff;
    delete[] dim;

  }

  void
  PosteriorHDF5Reader::GetParameters(const size_t idx, std::vector<fp_t>& parameters)
  {
    // Returns the iteration of the posterior specified by idx
#ifndef NDEBUG
    if(idx >= size_) throw range_exception("idx out of range");
#endif
    // Get parameters width
    parameters.resize(numParams_);
    parametersPt_->GetPacket(idx,parameters.data());
  }

  void
  PosteriorHDF5Reader::GetInfections(const size_t idx, std::vector<IPTuple_t>& infections)
  {
    // Returns the infection vector of the idx'th iteration of the posterior
#ifndef NDEBUG
    if(idx >= size_) throw range_exception("idx out of range");
#endif

    hvl_t buffer;
    infectionsPt_->GetPacket(idx, &buffer);
    IPTupleHack_t* tuples = (IPTupleHack_t*) buffer.p;
    infections.resize(buffer.len);
    for(size_t i=0; i<buffer.len; ++i) {
        infections[i].idx = tuples[i].idx;
        infections[i].val = tuples[i].val;
    }

    free(buffer.p);
  }

} /* namespace EpiRisk */
