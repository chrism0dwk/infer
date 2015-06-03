##########################################################################
## Copyright 2015 Chris Jewell                                          ##
##                                                                      ##
## This file is part of nztheileria.                                    ##
##                                                                      ##
## nztheileria is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by ##
## the Free Software Foundation, either version 3 of the License, or    ##
## (at your option) any later version.                                  ##
##                                                                      ##
## nztheileria is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of       ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        ##
## GNU General Public License for more details.                         ##
##                                                                      ##
## You should have received a copy of the GNU General Public License    ##
## along with nztheileria.  If not, see <http://www.gnu.org/licenses/>. ##
##########################################################################


## Model fitting algorithms

#' Fits the Theileria model of Jewell and Brown (2012) JRS Interface to data.
#'
#' MCMC algorithm for fitting dynamical SI model with latent vector surface to a vector-borne disease.
#'
#' @param pop a \code{data.frame} with column headings 'id','x','y','isdairy','region' for the individuals' labels, x and y coordinates (in Mercatorial projection), dairy farm or not, and the region the farm is located in.
#' @param epi a \code{data.frame} with column headings 'id','i','d' for the individuals' labels, (best guess) infection time, and detection time.
#' @param contact a \code{data.frame} representing a weighted edge list of a contact graph.  Headings must be 'from','to','weight' containing from and to labels, and a weight.
#' @param ticks a \code{data.frame} with headings 'region','numpos','total','a','b'.  Here, 'numpos' herds out of 'total' test positive for related vector-borne disease strains.  'a' and 'b' are the parameters of a Beta(a,b) distribution encoding prior belief about tick activity in each region.
#' @param init List containing MCMC starting values for parameters 'epsilon,delta,omega,beta1,beta2,alpha1,alpha2,alpha3,nu,zeta,b'
#' @param prior List of vectors of length 2 with hyperparameters for '(epsilon, delta, beta1, beta2, alpha1, alpha2, alpha3, b'
#' @param obsTime the 'observation' (or analysis) time of the epidemic.
#' @param control \code{list} of MCMC control settings, including \code{dlimit} (distance matrix limit), \code{n.iter} (number of MCMC iterations), \code{gpuid} (GPU device id for likelihood calculation), \code{seed} (PRNG seed), \code{tune.I} (MH tuning constant for infection times), \code{reps.I} (number of infection times to update per sweep of transmission parameters), \code{tmpdir} (path to temporary backing file for posterior output)
#' @return an object of type \code{Posterior}
#' @references Jewell CP, Brown RG (2015) Bayesian data assimilation provides rapid decision support for vector borne diseases. \emph{J. Roy. Soc. Interface} In press.
#' @author Chris Jewell \email{c.jewell@@lancaster.ac.uk}
mcmc <- function(pop, epi, contact, ticks, init, priors, obsTime=max(epi$n), control)
    {

    # Sanitize data
    if(any(names(pop) != c('id','x','y','isdairy','region')))
        stop('Malformed population data.  Must have columns "<id>,<x>,<y>,<isdairy>,<region>".')
    if(any(names(epi) != c('id','i','d')))
        stop('Malformed epidemic data. Must have columns "<id>,<i>,<d>".')
    if(any(names(contact) != c('from','to','weight')))
        stop('Malformed contact matrix edge list. Must have columns "<from>,<to>,<weight>".')
    if(any(names(ticks) != c('region','numpos','total','a','b')))
        stop('Malformed tick data.  Must have columns "<region>,<numpos>,<total>,<a>,<b>".')
    
    # Default initial parameters
    param.names <- list(epsilon=1e-6,delta=1,omega=1.2,beta1=0.05,beta2=0.05,alpha1=0.1,alpha2=0.1,alpha3=0.1,nu=19,zeta=0.5,b=0.1)

    # Set up initial values
    if(!is.list(init)) stop("List required for argument 'init'")
    if(any(!(names(param.names) %in% names(init))))
        stop("Malformed init.  Must contain names 'epsilon,delta,omega,beta1,beta2,alpha1,alpha2,alpha3,nu,zeta,b'")
    if(any(!sapply(init, is.numeric))) stop("Missing value(s) in 'init'")

    # Priors
    prior.names <- list(epsilon=c(2.7,1e8), delta=c(1,1), beta1=c(4,16000), beta2=c(2,2), alpha1=c(1,50), alpha2=c(1,50), alpha3=c(32,8), zeta=c(5,2), b=c(2.5,50))
    if(any(!(names(prior.names) %in% names(priors))))
        stop(paste("Malformed priors.  Must contain names",names(prior.names)))

    # Set up control list
    if(is.null(control$dlimit)) control$dlimit <- 25.0
    else control$dlimit <- as.numeric(control$dlimit)
    
    if(is.null(control$n.iter)) control$n.iter <- 10000
    else control$n.iter <- as.integer(control$n.iter)
    
    if(is.null(control$gpuid)) control$gpuid <- -1
    else control$gpuid <- as.integer(control$gpuid)
    
    if(is.null(control$seed)) control$seed <- as.integer(round((2^31-1) * runif(1)))
    else control$seed <- as.integer(control$seed)

    if(is.null(control$tune.I)) control$tune.I <- 0.1
    else control$tune.I <- as.numeric(control$tune.I)
    
    if(is.null(control$reps.I))
        control$reps.I <- ceiling(nrow(epi) * 0.1)
    else control$reps.I <- as.integer(control$reps.I)

    if(is.null(control$tmpdir)) control$tmpdir <- tempdir()
    else control$tmpdir <- as.character(control$tmpdir)

    outfile <- tempfile(pattern="theileria.posterior",tmpdir=control$tmpdir,fileext=".hd5")

    # Underlying generic code requires SINR model -- translate epidemics
    epi$r <- obsTime
    epi$type <- 'IP'
    names(epi) <- c('id','i','n','r','type')
    
    .Call("TheileriaMcmc", population=pop,
          epidemic=epi,
          contact=contact,
          ticks=ticks,
          obsTime=obsTime,
          init=init,
          priorParms=priors,
          control=control,
          outputfile=outfile)

    Posterior(outfile)
}
