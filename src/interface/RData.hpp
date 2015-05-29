// License here

#ifndef RDATA_HPP_
#define RDATA_HPP_

#include <Rcpp.h>

#include "types.hpp"
#include "Data.hpp"
#include "TheileriaData.hpp"
#include "Individual.hpp"
#include "DataImporter.hpp"

struct TickSurv
{
  int region;
  float numpos;
  float total;
  float a;
  float b;
};

class PopRImporter : public EpiRisk::DataImporter<TheilData>
{
private:
  Rcpp::DataFrame& data_;
  int rowptr_;
  Rcpp::CharacterVector id_;
  Rcpp::NumericVector x_;
  Rcpp::NumericVector y_;
  Rcpp::NumericVector isDairy_;
  Rcpp::NumericVector tla_;
public:
  PopRImporter(Rcpp::DataFrame& population);
  virtual void open() {};
  virtual void close() {};
  virtual Record next();
  virtual void reset();
};

class EpiRImporter : public EpiDataImporter
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


class ContactRImporter : public ContactDataImporter
{  
private:
  Rcpp::DataFrame& data_;
  int rowptr_;
  Rcpp::CharacterVector from_;
  Rcpp::CharacterVector to_;
  Rcpp::NumericVector val_;
public:
  ContactRImporter(Rcpp::DataFrame& contact);
  virtual void open() {};
  virtual void close() {};
  virtual Record next();
  virtual void reset();
};


#endif
