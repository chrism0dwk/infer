// License here


#include "RData.hpp"


PopRImporter::PopRImporter(Rcpp::DataFrame& population) 
  :  data_(population),rowptr_(0)
{
  id_ = data_["id"];
  x_  = data_["x"];
  y_  = data_["y"];
  for(int col=3; col<population.size(); ++col)
    sp_.push_back(data_[col]);
}

PopRImporter::Record
PopRImporter::next()
{
  if(rowptr_ == id_.size()) throw EpiRisk::fileEOF(); 

  Record tmp;
  tmp.id = id_[rowptr_];
  tmp.data.x = x_[rowptr_];
  tmp.data.y = y_[rowptr_];

  tmp.data.cattle = sp_[0][rowptr_];
  if(sp_.size() <= 2) tmp.data.pigs = sp_[1][rowptr_];
  if(sp_.size() <= 3) tmp.data.sheep = sp_[2][rowptr_];

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
