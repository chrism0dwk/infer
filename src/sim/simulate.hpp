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

#ifndef _nztheileria_SIMULATE_H
#define _nztheileria_SIMULATE_H

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

using namespace Rcpp;
using std::cerr; using std::endl;

namespace EpiRisk {



  class S4CMatrixView {
  private:
    IntegerVector colptr_;
    IntegerVector rowidx_;
    NumericVector val_;
    IntegerVector dim_;
 
  public:
    S4CMatrixView(S4 cmatrix) : colptr_(cmatrix.slot("p")), 
				rowidx_(cmatrix.slot("i")), 
				val_(cmatrix.slot("x")),
				dim_(cmatrix.slot("Dim"))
    {}
    
    double
    operator() (int row, int col) const
    {
      IntegerVector::const_iterator rowbegin = rowidx_.begin() + colptr_[col];
      IntegerVector::const_iterator rowend = rowidx_.begin() + colptr_[col+1];
      IntegerVector::const_iterator it = std::find(rowbegin,rowend,row);
      int valpos = it - rowidx_.begin();
      if(it != rowend) return val_[valpos];
      else return 0.0;
    }

    IntegerVector
    GetColPtr() const { return colptr_; }

    IntegerVector
    GetRowIdx() const { return rowidx_; }

    NumericVector
    GetVal() const { return val_; }

    IntegerVector
    GetDim() const { return dim_; }

    size_t
    GetNNZ() const { return val_.size(); }

  };

  
    
  class Simulator
    {
    public:
      Simulator(DataFrame population, S4 contact, NumericVector parameters, NumericVector dLimit);
      ~Simulator();
      void Gillespie(const double maxtime);
      DataFrame GetPopulation() const;
      
    private:
      
      typedef std::multimap<double, size_t> PressureCDF;
      typedef std::vector< std::pair<int,double> > PressureVector;
      typedef std::vector<size_t> IntPtrList;
      
      virtual double h(const size_t i, const double time) const;
      //virtual double H(const size_t i, const double time) const;
      virtual double K(const size_t i, const size_t j) const;
      virtual double beta(const size_t i, const size_t j) const;
      virtual double background() const;
      
      void Initialize();
      double GetMaxPressure() const;
      
      DataFrame population_;
      S4CMatrixView* contact_;
      NumericVector parameter_;
      NumericVector x_;
      NumericVector y_;
      NumericVector ticks_;
      NumericVector infecTime_;
      PressureCDF pressure_;
      IntPtrList infecList_;
      size_t popSize_;
      double time_;
      double dLimit_;
  };
  
}


RcppExport SEXP GetMatrixElement(SEXP contact, SEXP i, SEXP j);


RcppExport SEXP Simulate(SEXP population, SEXP contact, SEXP parameters, SEXP dLimit, SEXP maxtime);

#endif
