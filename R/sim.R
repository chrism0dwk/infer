#' Simulate a theileriosis SI epidemic
#'
#' Simulates an SI epidemic model of a vector-borne disease with latent vector surface.
#' 
#' @param pop a \code{data.frame} with column headings 'id','x','y','isdairy','region' for the individuals' labels, x and y coordinates (in Mercatorial projection), dairy farm or not, and the region the farm is located in.
#' @param epi a \code{data.frame} with column headings 'id','i','d' for the individuals' labels, (best guess) infection time, and detection time.
#' @param contact a \code{data.frame} representing a weighted edge list of a contact graph.  Headings must be 'from','to','weight' containing from and to labels, and a weight.
#' @param params a \code{list} of parameters with atomic 'epsilon','delta','omega','beta1','beta2','alpha1','alpha2','alpha3','nu','zeta','a','b', and a vector representing the 'phi's.  The \code{name} of each element of the \code{params@@phi} vector link the value to the correct region.
#' @param dlimit distance limit for spatial spread
#' @param maxtime the maximum time up to which to simulate.
simulate <- function(pop,epi,contact,params,dlimit=50,maxtime=Inf)
{
    require(Matrix)

                                        # Population
    if(any(names(pop) != c('id','x','y','isdairy','region')))
        stop('Malformed population data.  Must have columns "<id>,<x>,<y>,<isdairy>,<region>".')

                                        # Epidemic seeds
    if(any(names(epi) != c('id','i','d')))
        stop("Malformed epidemic list.  Should have fields <id>,<i>")
    pop$i <- rep(Inf, nrow(pop)) # Non-infectives have i==Inf
    pop$i[match(epi$id,pop$id)] <- epi$i

                                        # Contact matrix
    if(any(names(contact) != c('from','to','weight')))
        stop("Malformed contact matrix edge list.  Should have fields <from>,<to>,<weight>.")
    spContact <- sparseMatrix(i=match(contact$from,pop$id), j=match(contact$to,pop$id), x=contact$weight, dims=rep(nrow(pop),2))

                                        # Parameters
    param.names <- c('epsilon','delta','omega','beta1','beta2','alpha1','alpha2','alpha3','nu','zeta','a','b','phi')
    if(any(!(param.names %in% names(params))))
        stop("Missing required parameters 'epsilon','delta','omega','beta1','beta2','alpha1','alpha2','alpha3','nu','zeta','a','b','phi'")
    
    p<-NULL # Put parameters in below.  Zeta is dealt with outside the simulation.
    with(params, p<<-c(epsilon,beta1,beta2,0,nu,delta,alpha1,alpha2,alpha3))
    
    tlanums <- as.numeric(names(params$phi))
    pop$ticks<-params$phi[match(pop$tla,tlanums)]
    pop$ticks[pop$isdairy==1] <- pop$ticks[pop$isdairy==1] * params$zeta
    
    rtn <- .Call("Simulate",population=pop,contact=spContact,parameter=p,dLimit=dlimit,maxtime=maxtime)

    rtn <- rtn[rtn$i<Inf,]
    rtn$d <- rtn$i + rgamma(nrow(rtn),params$a,params$b)
    rtn[,c('id','i','d')]
}
