// GPLv3 Here

#ifndef _infer_SPSINRMCMC_CPP
#define _infer_SPSINRMCMC_CPP

#include <R.h>
#include <Rcpp.h>

RcppExport SEXP SpSINRMcmc(const SEXP population, 
			   const SEXP epidemic,
			   const SEXP obsTime,
			   const SEXP movtBan,
			   const SEXP init,
			   const SEXP priorParms,
			   const SEXP control,
			   const SEXP outputfile);


#endif
