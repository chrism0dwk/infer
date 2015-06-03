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
