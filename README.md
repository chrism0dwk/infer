README file for nztheileria package
====================================

Author: Chris Jewell <c.jewell@lancaster.ac.uk> (c) 2015
Created: 2015-06-01
Licence: GPLv3

This package contains code implementing the model for New Zealand 
*Theileria orientalis* (Ikeda) in cattle:

Jewell CP, Brown RG (2015) Bayesian data assimilation provides rapid decision support for vector borne diseases. *J. Roy. Soc. Interface. In press.*


Building
--------

This package depends on the following installed libraries

* Boost >= 1.55
* CUDA >= 5.0
* cudpp >= 2.2
* HDF5 >= 1.8.14
* R >= 3.1.0 with Rcpp >= 0.11.5

The R package configure file tries to guess the locations of these packages.  If libraries are in non-standard locations, supply the path(s) to the headers and libraries via CPPFLAGS and LDFLAGS, such as

     $ R CMD INSTALL --configure-vars="CPPFLAGS=-I/path/to/headers LDFLAGS=-L/path/to/libraries" nztheileria

