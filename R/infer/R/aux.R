# Auxilliary helper functions

crn <- function(n)
  {
    # Takes a number n, and returns a pair
    # of integers (x,y) describing a near-square
    # grid containing n + m elements, minimising
    # m.

    rows <- floor(sqrt(n))
    cols <- rows
    m <- rows*cols - n
    if(m == 0) return(as.integer(c(rows,cols)))

    m <- -Inf

    # Add columns until we have a positive remainder
    while(m < 0) {
      cols <- cols + 1
      m <- rows*cols - n
    }

    # Now adjust the aspect ratio to minimise m
    while(TRUE)
      {
        mprime <- (cols+1)*(rows-1) - n
        if(mprime < m & mprime >= 0 & rows/cols > 0.4)
          {
            cols <- cols + 1
            rows <- rows - 1
            m <- mprime
          }
        else break
      }

    d <- c(rows,cols)
    return(as.integer(d))
  }

