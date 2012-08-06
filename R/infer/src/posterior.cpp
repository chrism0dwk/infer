#include "posterior.hpp"

#include <iostream>
#include <cstdlib>
#include <stdexcept>

#include <H5Cpp.h>
#include <H5PacketTable.h>
#include <H5Exception.h>

#define PARAMPATH "posterior/parameters"
#define INFECPATH "posterior/infections"

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

void
postparams(H5::H5File& file, size_t burnin, size_t thin,
    Rcpp::DataFrame& posterior)
{

  char paramPath[] = "posterior/parameters";

  H5::DataSet pds = file.openDataSet(paramPath);

  // Get parameter names
  Rcpp::CharacterVector rTags;
  try
    {
      readTags(pds, rTags);
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

  // Now open parameters as a packet list
  FL_PacketTable parameters(file.getId(), paramPath);
  hsize_t totalRecords = parameters.GetPacketCount();
  hsize_t numFields = getFLPTwidth(pds);

  // Construct R data.frame
  hsize_t numRecords = (totalRecords - burnin) / thin;
  Rcpp::List dataList(numFields);
  for (int i = 0; i < numFields; ++i)
    {
      Rcpp::NumericVector myVector(numRecords);
      dataList[i] = myVector;
    }
  dataList.attr("names") = rTags;

  // Copy in data
  float* record = new float[numFields];
  //parameters.SetIndex(_burnin[0]);
  for (int i = 0; i < numRecords; ++i)
    {
      parameters.GetPacket(burnin + i * thin, record);
      for (int j = 0; j < numFields; ++j)
        {
          Rcpp::NumericVector col = dataList[j];
          col[i] = record[j];
        }
    }
  delete[] record;

  posterior = Rcpp::DataFrame(dataList);
}

void
postinfecs(H5::H5File& file, size_t burnin, size_t thin, Rcpp::List& posterior)
{
  char infecpath[] = INFECPATH;

  FL_PacketTable infections(file.getId(), infecpath);

  size_t totalRecords = infections.GetPacketCount();
  size_t numRecords = (totalRecords - burnin) / thin;

  Rcpp::List data(numRecords);

  hvl_t buff;
  for (size_t i = 0; i < numRecords; ++i)
    {
      infections.GetPacket(burnin + i * thin, &buff);

      Rcpp::IntegerVector ids(buff.len);
      Rcpp::NumericVector val(buff.len);
      ipTuple_t* records = (ipTuple_t*) buff.p;

      for (size_t j = 0; j < buff.len; ++j)
        {
          ids[j] = records[j].idx;
          val[j] = records[j].val;
        }

      free(buff.p);
      Rcpp::DataFrame valframe(Rcpp::DataFrame::create(Rcpp::Named("time") =
          val));
      valframe.attr("row.names") = ids;

      data[i] = valframe;
    }

  posterior = data;
}

RcppExport SEXP
readPosterior(SEXP filename, SEXP burnin, SEXP thin, SEXP actions)
{
  Rcpp::IntegerVector _actions(actions);
  Rcpp::CharacterVector _filename(filename);
  Rcpp::IntegerVector _burnin(burnin);
  Rcpp::IntegerVector _thin(thin);

  try
    {
      H5::H5File file(_filename[0], H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);

      Rcpp::List posterior(2);
      Rcpp::CharacterVector posteriorTags(2);
      posteriorTags[0] = "parameters";
      posteriorTags[1] = "infec";
      posterior.attr("names") = posteriorTags;

      Rcpp::DataFrame params;
      Rcpp::List infecs;

      // Get parameters
      if (_actions[0] == 0)
        {
          Rcpp::DataFrame params;
          postparams(file, _burnin[0], _thin[0], params);
          Rcpp::List infecs;
          postinfecs(file, _burnin[0], _thin[0], infecs);
          return Rcpp::List::create(Rcpp::Named("parameters") = params,
              Rcpp::Named("infec") = infecs);
        }
      else if (_actions[0] == 1)
        {
          Rcpp::DataFrame params;
          postparams(file, _burnin[0], _thin[0], params);
          return params;
        }
      else if (_actions[0] == 2)
        {
          Rcpp::List infecs;
          postinfecs(file, _burnin[0], _thin[0], infecs);
          return infecs;
        }
      else
        {
          throw std::runtime_error("Invalid action specified");
        }
    }
  catch (H5::Exception& e)
    {
      ::Rf_error("Error opening HDF5 file");
    }
  catch (std::runtime_error& e)
    {
      ::Rf_error(e.what());
    }
  catch (...)
    {
      ::Rf_error("Unknown exception reading file");
    }
}
