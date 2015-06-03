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


#include "RData.hpp"


PopRImporter::PopRImporter(Rcpp::DataFrame& population) 
  :  data_(population),rowptr_(0)
{
  id_ = data_["id"];
  x_  = data_["x"];
  y_  = data_["y"];
  isDairy_ = data_["isdairy"];
  tla_ = data_["tla"];
}

PopRImporter::Record
PopRImporter::next()
{
  if(rowptr_ == id_.size()) throw EpiRisk::fileEOF(); 

  Record tmp;
  tmp.id = id_[rowptr_];
  tmp.data.x = x_[rowptr_];
  tmp.data.y = y_[rowptr_];
  tmp.data.isDairy = isDairy_[rowptr_];
  tmp.data.ticks = tla_[rowptr_];

  rowptr_++;
  
  return tmp;
}

void
PopRImporter::reset()
{
  rowptr_ = 0;
}


EpiRImporter::EpiRImporter(Rcpp::DataFrame& epidemic) : data_(epidemic),rowptr_(0)
{
  id_ = data_["id"];
  i_ = data_["i"];
  n_ = data_["n"];
  r_ = data_["r"];
  type_ = data_["type"];
}


EpiRImporter::Record
EpiRImporter::next()
{
  if(rowptr_ == id_.size()) throw EpiRisk::fileEOF();

  Record record;

  record.id = id_[rowptr_];
  record.data.I = i_[rowptr_];
  record.data.N = n_[rowptr_];
  record.data.R = r_[rowptr_];
  record.data.type = type_[rowptr_];

  rowptr_++;

  return record;
}


void
EpiRImporter::reset()
{
  rowptr_ = 0;
}

ContactRImporter::ContactRImporter(Rcpp::DataFrame& contact) : data_(contact), rowptr_(0)
{
  from_ = data_["from"];
  to_ = data_["to"];
  val_ = data_["weight"];
}

ContactRImporter::Record
ContactRImporter::next()
{
  if(rowptr_ == from_.size()) throw EpiRisk::fileEOF();
  Record record;
  record.id = std::string(from_[rowptr_]);
  record.data.i = std::string(from_[rowptr_]);
  record.data.j = std::string(to_[rowptr_]);
  record.data.val = val_[rowptr_];
  rowptr_++;
  return record;
}

void
ContactRImporter::reset()
{
  rowptr_ = 0;
}
