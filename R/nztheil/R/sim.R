


.initSimMethods <- function()
  {


setGeneric("berp.sim", function(model, control, params) standardGeneric("berp.sim"))
setMethod("berp.sim","SpatPointSINR", function(model, control, params) {

  if(missing(control)) control <- list(algorithm="Gillespie")
  else if(is.null(control$algorithm)) control$algorithm <- "Gillespie"

  simepi <- NULL
  
  if(control$algorithm == "Gillespie") {

    if(is.null(control$mintime)) control$mintime <- model@obsTime
    else control$mintime <- as.numeric(control$mintime)

    if(is.null(control$maxtime)) control$maxtime <- Inf
    else control$maxtime <- as.numeric(control$maxtime)

    if(is.null(control$sim.censored.events)) control$sim.censored.events <- TRUE
    else control$sim.censored.events <- as.logical(control$sim.censored.events)

    if(is.null(control$ntor)) control$ntor <- 1.0
    else control$ntor <- as.numeric(control$ntor)

    if(is.null(control$seed)) control$seed <- as.integer(round((2^31-1)*runif(1)))
    else control$seed <- as.integer(control$seed)

    simepi <- .Call("SpPointSINRSim", population=model@population,
                    epidemic=model@epidemic,
                    obstime=model@obsTime,
                    movtBan=model@movtBan,
                    params=params,
                    control=control)
  }
  else stop("invalid algorithm type")

  SpatPointSINR(pop, simepi)
})

setMethod("berp.sim","NZTheileriaSI", function(model, control, params) {

  if(missing(control)) control <- list(algorithm="Gillespie")
  else if(is.null(control$algorithm)) control$algorithm <- "Gillespie"

  simepi <- NULL

  if(control$algorithm == "Gillespie") {

    if(is.null(control$mintime)) control$mintime <- model@obsTime
    else control$mintime <- as.numeric(control$mintime)

    simepi <- .Call("NZTheileriaSim", population=model@population,
                    contact=model@contact,
                    obstime=model@obsTime,
                    params=params,
                    control=control)
  }
  else stop("invalid algorithm type")

  simepi
})



















  }
