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

#ifndef THEILERIAMCMC_HPP
#define THEILERIAMCMC_HPP

#include <R.h>
#include <Rcpp.h>

RcppExport SEXP TheileriaMcmc(const SEXP population,
			      const SEXP epidemic,
			      const SEXP contact,
			      const SEXP ticks,
			      const SEXP obsTime,
			      const SEXP init,
			      const SEXP priorParms,
			      const SEXP control,
			      const SEXP outputfile);

#endif

