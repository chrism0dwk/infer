# Model fitting algorithms

.initFittingMethods <- function() {

setGeneric("berp.fit", function(model,control,...) standardGeneric("berp.fit"))

setMethod("berp.fit", "SpatPointSINR", function(model, control, init)
          {

            if(missing(control)) control <- list(algorithm="mcmc")
            else if (is.null(control$algorithm)) control$algorithm = "mcmc"


            if(control$algorithm == "mcmc") {
              # Set up initial values
              if(!is.list(init)) stop("List required for argument 'init'")
              if(any(is.na(match(model@param.names, names(init))))) stop("Malformed initial values list")
              if(any(!sapply(init, is.numeric))) stop("Missing value(s) in 'init'")

              # Set up control list
              if(is.null(control$dlimit)) control$dlimit <- 25000.0
              else control$dlimit <- as.numeric(control$dlimit)
               
              if(is.null(control$n.iter)) control$n.iter <- 10000
              else control$n.iter <- as.integer(control$n.iter)
              
              if(is.null(control$movtban)) control$movtban <- FALSE
              else control$movtban <- as.logical(control$movtban)
              
              if(is.null(control$gpuid)) control$gpuid <- -1
              else control$gpuid <- as.integer(control$gpuid)
            
              if(is.null(control$occults)) control$occults <- "no"
              else if(!(as.character(control$occults) %in% c("yes","no","dconly")))
                stop("Invalid option in control$occults.  Options are 'yes','no','dconly'.")
              else control$occults <- as.character(control$occults)

              if(is.null(control$powers)) control$powers <- TRUE
              else control$powers <- as.logical(control$powers)
              
              if(is.null(control$seed)) control$seed <- as.integer(round((2^31-1) * runif(1)))
              else control$seed <- as.integer(control$seed)

              if(is.null(control$ncratio)) control$ncratio <- 0.3
              else control$ncratio <- as.numeric(control$ncratio)
              
              if(is.null(control$tune.I)) control$tune.I <- 0.1
              else control$tune.I <- as.numeric(control$tune.I)
              
              if(is.null(control$reps.I))
                control$reps.I <- ceiling(sum(model@epidemic$type == 'IP') * 0.1)
              else control$reps.I <- as.integer(control$reps.I)

              if(is.null(control$infer.latent.period.scale)) control$infer.latent.period.scale <- FALSE
              else control$infer.latent.period.scale <- as.logical(control$infer.latent.period.scale)
              
              if(is.null(control$tmpdir)) control$tmpdir <- tempdir()
              else control$tmpdir <- as.character(control$tmpdir)

              outfile <- tempfile(pattern="berp.posterior",tmpdir=control$tmpdir,fileext=".hd5")
              cat("Before call\n")
              .Call("SpSINRMcmc", population=model@population,
                    epidemic=model@epidemic,
                    obsTime=model@obsTime,
                    movtBan=model@movtBan,
                    init=init,
                    priorParms=model@prior,
                    control=control,
                    outputfile=outfile)
              cat("After call\n")
            }
            else stop("invalid algorithm type!")
            
            #Posterior(outfile)
          }
)


}
