# These functions are to get the leave one out score for BART by estimation.
bart.loo = function(object, 
                    y.train, 
                    probit, 
                    n.train, 
                    nskip, 
                    ndpost, 
                    keepevery) {
  
  #--------------------------------
  # Prepare likelihood matrix
  if (probit) {
    ## likelihood matrix: ndpost x n
    lik.mat = t(apply(object$prob.train, 1, function(s) s * y.train + (1 - s) * (1 - y.train)))
    
  } else {
    ## y.hat.posterior.samples: ndpost x n
    y.hat.posterior.samples = object$yhat.train

    raw.sigma = object$sigma[-c(1:nskip)]
    sigma = c()
    for (i in 1:length(raw.sigma)) {
      if (((i-1) %% keepevery) == 0) 
        sigma = c(sigma, raw.sigma[i])
    }

    ## likelihood matrix: ndpost x n
    lik.mat = matrix(NA, nrow = ndpost, ncol = n.train)
    for (i in 1:ndpost) {
      for (j in 1: n.train) {
        lik.mat[i, j] = dnorm(x = y.train[j], mean = y.hat.posterior.samples[i, j], sd = sigma[i])
      }
    }

    rm(list = c("y.hat.posterior.samples", "raw.sigma", "sigma"))
  }

  ## log-likelihood matrix: ndpost x n
  llik.mat = log(lik.mat)

  #--------------------------------
  # Compute relative effective sample size
  chain.id = rep(1, ndpost)
  r.eff = relative_eff(lik.mat, chain.id)
  
  #--------------------------------
  # Get PSIS-LOO result
  loo.result = loo(x = llik.mat, r_eff = r.eff)

  rm(list = c("llik.mat", "lik.mat"))
  gc()

  return(loo.result)
}
