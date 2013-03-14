# Declares the model classes





.initClasses <- function() {
setClass("EpiPopModel", representation(id="character"))



##################################
## The Spatial Point SINR model ##
##################################
setClass("SpatPointSINR", contains="EpiPopModel", representation(population="data.frame", epidemic="data.frame", prior="list", obsTime="numeric", movtBan="numeric", param.names="character"))
setMethod("initialize","SpatPointSINR", function(.Object, population, epidemic, obsTime=NULL, movtBan=NULL) {


  # Sanity checks on population
  if(ncol(population) < 4) stop("Unexpected number of columns in 'population' (should be >= 4)")
  if(any(c("id","x","y") != names(population[,1:3]))) stop("Could not find at least one of 'id','x',or 'y'.  Check 'population' fields.")
  
  # id's, and x/y coords are straightforward
  population$id = as.character(population$id)
  population$x = as.numeric(population$x)  # Distances expressed in map units
  population$y = as.numeric(population$y)

  # Sanitise number of animals
  animals <- population[,4:ncol(population), drop=F]
  if(ncol(animals) > 3) stop("Number of cols in animals > 3")
  if(any(animals<0)) stop("Negative animals found")
  if(any(is.na(animals))) stop("NA found in animals")
  for(i in 1:ncol(animals)) animals[,i] <- animals[,i] / mean(animals[,i])
  population[,4:ncol(population)] <- animals
  
  .Object@population = population
  
  # Sanitise epidemic data
  if(ncol(epidemic) != 5)
    stop("Malformed epidemic data.frame.  Check data.frame(id,i,n,r,type)")
  if(any(c('id','i','n','r','type') != names(epidemic)))
    stop("Could not identify columns in 'epidemic'.  I require names(epidemic) <- c('id','i','n','r','type')")
  epidemic$id <- as.character(epidemic$id)
  if(any(!(epidemic$id %in% .Object@population$id)))
    stop("Unidentified id in epidemic")
  if(any(epidemic$i[epidemic$type == 'IP'] > epidemic$n[epidemic$type == 'IP']))
    stop("Detection time > Infection time for an IP")
  if(any(epidemic$n > epidemic$r))
    stop("Removal time < Detection time in epidemic")
  if(any(!(epidemic$type %in% c("IP","DC"))))
    stop("Unrecognised infection type.  Valid types are 'IP' or 'DC'.")
  
  # Sanitize observation time and movement ban
  if(is.null(obsTime)) obsTime <- max(epidemic$n, epidemic$r)
  if(is.null(movtBan)) movtBan <- obsTime
  if(any(epidemic$n > obsTime) | any(epidemic$r > obsTime)) stop("Detection or Removal times > obsTime should be set to Inf")
 
  .Object@epidemic <- epidemic
  .Object@obsTime <- obsTime
  .Object@movtBan <- movtBan

  # Set up the default prior
  .Object@param.names <- c("epsilon1","epsilon2","gamma1","gamma2","xi_2","xi_3","psi_1","psi_2","psi_3","zeta_2","zeta_3","phi_1","phi_2","phi_3","delta","a","b","alpha")
  .Object@prior <- list(epsilon1=c(1,1), epsilon2=c(1,1), gamma1=c(1,1), gamma2=c(1,1), xi_2=c(1,1), xi_3=c(1,1), psi_1=c(2,2), psi_2=c(2,2), psi_3=c(2,2), zeta_2=c(1,1), zeta_3=c(1,1), phi_1=c(2,2), phi_2=c(2,2), phi_3=c(2,2), delta=c(1,1), b=c(1,1))
  
  .Object
} )





# Set methods for the model objects
setGeneric("priors", function(object) standardGeneric("priors"))
setMethod("priors","SpatPointSINR", function(object) object@prior)

setGeneric("priors<-", function(obj, value) standardGeneric("priors<-"))
setReplaceMethod("priors","SpatPointSINR", function (obj, value) {

  if(any(is.na(match(obj@param.name, names(value))))) stop("Mis-specified prior list")
      
  canCoerce <- sapply(value, function(x) ifelse(is.numeric(x) & length(x)==2, TRUE, FALSE))
  if(any(canCoerce == FALSE)) stop("Mis-specified prior")
  
  obj@prior <- value
  obj
})


          
} # .initClasses



SpatPointSINR <- function(population, epidemic, priors, obsTime=NULL, movtBan=NULL)
  {
    obj <- new("SpatPointSINR", population=population, epidemic=epidemic, obsTime, movtBan)
    if(!missing(priors)) priors(obj) <- priors

    obj
  }
