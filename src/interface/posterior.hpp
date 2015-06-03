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

#ifndef _infer_POSTERIOR_CPP
#define _infer_POSTERIOR_CPP

#include <R.h>
#include <Rcpp.h>

RcppExport SEXP getPosteriorParams(SEXP filename,
			SEXP rows,
			SEXP cols);

RcppExport SEXP getPosteriorInfecs(SEXP filename,
                                   SEXP rows,
                                   SEXP cols);

RcppExport SEXP getPosteriorParamInfo(SEXP filename);

RcppExport SEXP getPosteriorParamSummary(SEXP filaname);

RcppExport SEXP getPosteriorInfecInfo(SEXP filename);

RcppExport SEXP getPosteriorInfecSummary(SEXP filename);

RcppExport SEXP getPosteriorLen(SEXP filename);

RcppExport SEXP getPosteriorModel(SEXP filename);


#endif
