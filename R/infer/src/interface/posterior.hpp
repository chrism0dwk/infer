// Put GPLv3 here

#ifndef _infer_POSTERIOR_CPP
#define _infer_POSTERIOR_CPP

//#include <R.h>
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
