# GPLv3 here

read.posterior <- function(filename, burnin, thin, actions="both")
{

   action <- switch(actions, both=0, parameters=1, infections=2, stop("Invalid parameter value 'actions'"))

   return(.Call("readPosterior", filename, burnin, thin, action))
}

