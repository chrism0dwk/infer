// License here

#ifndef RDATA_HPP_
#define RDATA_HPP_

#include <Rcpp.h>

#include "types.hpp"
#include "Data.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"

class PopRImporter : public EpiRisk::PopDataImporter
{
private:
  Rcpp::DataFrame& data_;
  int rowptr_;
  Rcpp::CharacterVector id_;
  Rcpp::NumericVector x_;
  Rcpp::NumericVector y_;
  std::vector<Rcpp::NumericVector> sp_;

public:
  PopRImporter(Rcpp::DataFrame& population);
  virtual void open() {};
  virtual void close() {};
  virtual Record next();
  virtual void reset();
};


class EpiRImporter : public EpiRisk::EpiDataImporter 
{
private:
  Rcpp::DataFrame& data_;
  int rowptr_;
  Rcpp::CharacterVector id_;
  Rcpp::NumericVector i_;
  Rcpp::NumericVector n_;
  Rcpp::NumericVector r_;
  Rcpp::CharacterVector type_;

public:
  EpiRImporter(Rcpp::DataFrame& epidemic);
  virtual void open() {};
  virtual void close() {};
  virtual Record next();
  virtual void reset();
};



#endif
