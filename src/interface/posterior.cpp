//////////////////////////////////////////////////////////////////////////
// Copyright 2015 Chris Jewell                                          //
// 								        //
// This file is part of nztheileria.                                    //
//                                                                      //
// nztheileria is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by //
// the Free Software Foundation, either version 3 of the License, or    //
// (at your option) any later version.                                  //
//                                                                      //
// nztheileria is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of       //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        //
// GNU General Public License for more details.                         //
//                                             			        //
// You should have received a copy of the GNU General Public License    //
// along with nztheileria.  If not, see <http://www.gnu.org/licenses/>. //
//////////////////////////////////////////////////////////////////////////

#include "posterior.hpp"

#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include <H5Cpp.h>
#include <H5PacketTable.h>
#include <H5Exception.h>

static char paramPath[] = "posterior/parameters";
static char infecpath[] = "posterior/infections";
static char idsPath[] = "posterior/ids";

typedef struct
{
  int idx;
  float val;
} ipTuple_t;

void
readTags(H5::DataSet& dataset, Rcpp::CharacterVector& rTags)
{

  H5::Attribute tags;
  tags = dataset.openAttribute("tags");

  H5::DataSpace tagDS = tags.getSpace();
  hsize_t* aDim = new hsize_t[tagDS.getSimpleExtentNdims()];
  tagDS.getSimpleExtentDims(aDim);

  char** readBuff = new char*[aDim[0]];
  H5::DataType paramTag_t = tags.getDataType();

  tags.read(paramTag_t, readBuff);
  for (int i = 0; i < aDim[0]; ++i)
    {
      rTags.push_back(readBuff[i]);
    }

  H5Dvlen_reclaim(paramTag_t.getId(), tagDS.getId(), H5P_DEFAULT, readBuff);
  delete[] readBuff;
  delete[] aDim;
}

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

RcppExport SEXP
getPosteriorParams(SEXP filename, SEXP rows, SEXP cols)
{
  // Rows and cols are 0-based!
  // No subscript boundary checking is performed.

  Rcpp::CharacterVector _filename(filename);
  Rcpp::IntegerVector _rows(rows);
  Rcpp::IntegerVector _cols(cols);

  Rcpp::List dataList(_cols.length());
  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);

      H5::DataSet pds = file.openDataSet(paramPath);

      // Open parameters as a packet list
      FL_PacketTable parameters(file.getId(), paramPath);
      hsize_t totalRecords = parameters.GetPacketCount();
      hsize_t numFields = getFLPTwidth(pds);

      // Get parameter names
      Rcpp::CharacterVector rTags;
      Rcpp::CharacterVector allTags;
      readTags(pds, allTags);
      for (size_t i = 0; i < _cols.length(); ++i)
        {
          rTags.push_back(allTags[_cols[i]]);
        }

      // Construct R data.frame
      for (int i = 0; i < _cols.length(); ++i)
        {
          Rcpp::NumericVector myVector(_rows.length());
          dataList[i] = myVector;
        }
      dataList.attr("names") = rTags;

      // Copy in data
      float* record = new float[numFields];
      for (int i = 0; i < _rows.length(); ++i)
        {
          parameters.GetPacket(_rows[i], record);
          for (int j = 0; j < _cols.length(); ++j)
            {
              Rcpp::NumericVector col = dataList[j];
              col[i] = record[_cols[j]];
            }
        }
      delete[] record;

      file.close();

    }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }

  if(_cols.length() == 1) return dataList[0];
  else return Rcpp::DataFrame(dataList);
}

RcppExport SEXP
getPosteriorInfecs(SEXP filename, SEXP rows, SEXP cols)
{
  // Rows and cols are 0-based!
  // No subscript boundary checking is performed
  // cols is currently unused

  Rcpp::CharacterVector _filename(filename);
  Rcpp::IntegerVector _rows(rows);

  Rcpp::List data(_rows.length());

  Rcpp::List info = getPosteriorInfecInfo(filename);
  Rcpp::CharacterVector tags = info[1];

  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);

      FL_PacketTable infections(file.getId(), infecpath);

      size_t totalRecords = infections.GetPacketCount();

      hvl_t buff;
      for (size_t i = 0; i < _rows.length(); ++i)
        {
          infections.GetPacket(_rows[i], &buff);

          Rcpp::CharacterVector ids(buff.len);
          Rcpp::NumericVector val(buff.len);
          ipTuple_t* records = (ipTuple_t*) buff.p;

          for (size_t j = 0; j < buff.len; ++j)
            {
              ids[j] = tags[records[j].idx];
              val[j] = records[j].val;
            }
          free(buff.p);

          val.attr("names") = ids;
          data[i] = val;
        }

      file.close();
    }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }

  return data;

}


RcppExport SEXP
getPosteriorParamInfo(SEXP filename)
{
  Rcpp::CharacterVector _filename(filename);

  Rcpp::List info(2);
  Rcpp::CharacterVector infoNames(2);
  infoNames[0] = "length";
  infoNames[1] = "tags";
  info.attr("names") = infoNames;

  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);
      FL_PacketTable parameters(file.getId(), paramPath);

      // Extract length
      Rcpp::NumericVector length(1);
      length[0] = parameters.GetPacketCount();
      info[0] = length;

      // Extract tags
      Rcpp::CharacterVector tags;
      H5::DataSet pds = file.openDataSet(paramPath);
      readTags(pds, tags);
      info[1] = tags;
      file.close();
    }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }

  return info;
}


RcppExport SEXP
getPosteriorInfecInfo(SEXP filename)
{
  Rcpp::CharacterVector _filename(filename);

  Rcpp::List info(2);
  Rcpp::CharacterVector infoNames(2);
  infoNames[0] = "length";
  infoNames[1] = "tags";
  info.attr("names") = infoNames;

  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);
      FL_PacketTable infecs(file.getId(), infecpath);

      // Extract length
      Rcpp::NumericVector length(1);
      length[0] = infecs.GetPacketCount();
      info[0] = length;

      // Extract tags
      H5::DataSet pds = file.openDataSet(idsPath);

      H5::DataSpace tagDS = pds.getSpace();
      hsize_t* aDim = new hsize_t[tagDS.getSimpleExtentNdims()];
      tagDS.getSimpleExtentDims(aDim);


      char** readBuff = new char*[aDim[0]];
      H5::DataType paramTag_t = pds.getDataType();
      pds.read(readBuff, paramTag_t);

      Rcpp::CharacterVector rTags(aDim[0]);
      for (int i = 0; i < aDim[0]; ++i)
        {
          rTags[i] = readBuff[i];
        }

      info[1] = rTags;

      H5Dvlen_reclaim(paramTag_t.getId(), tagDS.getId(), H5P_DEFAULT, readBuff);
      delete[] readBuff;
      delete[] aDim;





      file.close();
    }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }

  return info;
}


RcppExport SEXP
getPosteriorLen(SEXP filename)
{
  Rcpp::CharacterVector _filename(filename);

  Rcpp::NumericVector postLength(1);

  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);
      FL_PacketTable parameters(file.getId(), paramPath);
      postLength[0] = parameters.GetPacketCount();
      file.close();
    }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }

  return postLength;
}

RcppExport SEXP
getPosteriorModel(SEXP filename)
{
  Rcpp::CharacterVector _filename(filename);

  Rcpp::CharacterVector model(1);

  model[0] = "To be implemented";

  return model;
}


