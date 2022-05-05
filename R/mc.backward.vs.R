single.model.evaluation = function(x.train, 
                                   y.train, 
                                   x.test, 
                                   y.test, 
                                   n.train, 
                                   n.test, 
                                   n, 
                                   probit,
                                   xinfo, 
                                   numcut, 
                                   usequants, 
                                   cont, 
                                   rm.const, 
                                   k, 
                                   power, 
                                   base,
                                   split.prob,
                                   ntree, 
                                   ndpost, 
                                   nskip, 
                                   keepevery, 
                                   printevery,
                                   verbose) {
  if(probit) {
    bart = pbart(x.train = x.train, y.train = y.train, x.test = x.test, sparse = FALSE,
                 xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                 k = k, power = power, base = base, split.prob = split.prob,
                 ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                 printevery = printevery, verbose = verbose)
    
    ## model error: mean logarithmic loss
    prob.test = bart$prob.test.mean
    model.error = 0
    for (i in 1:n.test) {
      if((prob.test[i] == 0 & y.test[i] == 0) | (prob.test[i] == 1 & y.test[i] == 1)){
        model.error = model.error
      } else {
        model.error = model.error + y.test[i] * log(prob.test[i]) + (1 - y.test[i]) * log(1 - prob.test[i])
      }
    }
    model.error = - model.error / n.test
  } else {
    bart = wbart(x.train = x.train, y.train = y.train, x.test = x.test, sparse = FALSE,
                 xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                 k = k, power = power, base = base, split.prob = split.prob,
                 ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                 printevery = printevery, verbose = verbose)
    
    ## model error
    model.error = mean((bart$yhat.test.mean - y.test)^2)
  }
  
  # elpd.loo0
  loo.result = bart.loo(object = bart, y.train = y.train, probit = probit, n.train = n.train, 
                        nskip = nskip, ndpost = ndpost, keepevery = keepevery)
  elpd.loo = loo.result$estimates[1, 1]
  badK = sum(loo.result$diagnostics$pareto_k >= 0.7) / n.train
  
  res = list()
  res$model.error = model.error
  res$elpd.loo = elpd.loo
  res$badK = badK
  
  return(res)
}

multiple.models.evaluation = function(x.train, 
                                      y.train, 
                                      x.test, 
                                      y.test, 
                                      models,
                                      n.train, 
                                      n.test, 
                                      n, 
                                      probit,
                                      xinfo, 
                                      numcut, 
                                      usequants, 
                                      cont, 
                                      rm.const, 
                                      k, 
                                      power, 
                                      base,
                                      split.prob,
                                      ntree, 
                                      ndpost, 
                                      nskip, 
                                      keepevery, 
                                      printevery,
                                      verbose) {
  
  if(is.vector(models)) {
    res = matrix(NA, nrow = 1, ncol = 3)
    models = matrix(models, nrow = 1, ncol = length(models))
  } else {
    res = matrix(NA, nrow = nrow(models), ncol = 3)
  }
    
  for (i in 1:nrow(models)) {
    if(probit) {
      bart = pbart(x.train = x.train[, models[i, ], drop = FALSE], y.train = y.train, 
                   x.test = x.test[, models[i, ], drop = FALSE], 
                   sparse = FALSE, xinfo = xinfo, numcut = numcut, usequants = usequants, cont =cont, 
                   rm.const = rm.const, k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                   printevery = printevery, verbose = verbose)
      
      ## model error: mean logarithmic loss
      prob.test = bart$prob.test.mean
      res[i, 1] = 0
      for (j in 1:n.test) {
        if((prob.test[j] == 0 & y.test[j] == 0) | (prob.test[j] == 1 & y.test[j] == 1)){
          res[i, 1] = res[i, 1]
        } else {
          res[i, 1] = res[i, 1] + y.test[j] * log(prob.test[j]) + (1 - y.test[j]) * log(1 - prob.test[j])
        }
      }
      res[i, 1] = - res[i, 1] / n.test
    } else {
      bart = wbart(x.train = x.train[, models[i, ], drop = FALSE], y.train = y.train, 
                   x.test = x.test[, models[i, ], drop = FALSE], 
                   sparse = FALSE, xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, 
                   rm.const = rm.const, k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                   printevery = printevery, verbose = verbose)
      
      ## model error
      res[i, 1] = mean((bart$yhat.test.mean - y.test) ^ 2)
    }
    
    # elpd.loo0
    loo.result = bart.loo(object = bart, y.train = y.train, probit = probit, n.train = n.train, 
                          nskip = nskip, ndpost = ndpost, keepevery = keepevery)
    res[i, 2] = loo.result$estimates[1, 1]
    res[i, 3] = sum(loo.result$diagnostics$pareto_k >= 0.7) / n.train
  }
  
  result = list()
  result$res = res
  result$models = models
  return(result)
}

#' Backward selection with two filters (using parallel computation)
#' 
#' This function implements the backward variable selection approach for BART (see Algorithm 2 in Luo and Daniels (2021) for 
#' details). Parallel computation is used within each step of the backward selection approach.
#' 
#' The backward selection starts with the full model with all the predictors, followed by comparing the deletion of each predictor
#' using mean squared error (MSE) if the response variable is continuous (or mean log loss (MLL) if the response variable is binary)
#' and then deleting the predictor whose loss gives the smallest MSE (or MLL). This process is repeated until there is only one
#' predictor in the model and ultimately returns \code{ncol{x}} "winner" models with different model sizes ranging from \eqn{1} to
#' \code{ncol{x}}.\cr
#' Given the \code{ncol{x}} "winner" models, the one with the largest expected log pointwise predictive density based on leave-one-out 
#' (LOO) cross validation is the best model. See Section 3.3 in Luo and Daniels (2021) for details.\cr
#' If \code{true.idx} is provided, the precision, recall and F1 scores are returned.\cr
#' 
#' @param x A matrix or a data frame of predictors values with each row corresponding to an observation and each column 
#' corresponding to a predictor. If a predictor is a factor with \eqn{q} levels in a data frame, it is replaced with \eqn{q} dummy 
#' variables.
#' @param y A vector of response (continuous or binary) values.
#' @param split.ratio A number between \eqn{0} and \eqn{1}; the data set \code{(x, y)} is split into a training set and a testing
#' set according to the \code{split.ratio}.
#' @param probit A Boolean argument indicating whether the response variable is binary or continuous; \code{probit=FALSE} (by default)
#' means that the response variable is continuous. 
#' @param true.idx (Optional) A vector of indices of the true relevant predictors; if provided, metrics including precision, recall
#' and F1 score are returned.
#' @param xinfo A matrix of cut-points with each row corresponding to a predictor and each column corresponding to a cut-point.
#' \code{xinfo=matrix(0.0,0,0)} indicates the cut-points are specified by BART.
#' @param numcut The number of possible cut-points; If a single number is given, this is used for all predictors; 
#' Otherwise a vector with length equal to \code{ncol(x)} is required, where the \eqn{i-}th element gives the number of 
#' cut-points for the \eqn{i-}th predictor in \code{x}. If \code{usequants=FALSE}, \code{numcut} equally spaced 
#' cut-points are used to cover the range of values in the corresponding column of \code{x}. 
#' If \code{usequants=TRUE}, then min(\code{numcut}, the number of unique values in the corresponding column of 
#' \code{x} - 1) cut-point values are used.
#' @param usequants A Boolean argument indicating how the cut-points in \code{xinfo} are generated; 
#' If \code{usequants=TRUE}, uniform quantiles are used for the cut-points; Otherwise, the cut-points are generated uniformly.
#' @param cont A Boolean argument indicating whether to assume all predictors are continuous.
#' @param rm.const A Boolean argument indicating whether to remove constant predictors.
#' @param k The number of prior standard deviations that \eqn{E(Y|x) = f(x)} is away from \eqn{+/-.5}. The response 
#' (\code{y}) is internally scaled to the range from \eqn{-.5} to \eqn{.5}. The bigger \code{k} is, the more conservative 
#' the fitting will be.
#' @param power The power parameter of the polynomial splitting probability for the tree prior. Only used if 
#' \code{split.prob="polynomial"}.
#' @param base The base parameter of the polynomial splitting probability for the tree prior if \code{split.prob="polynomial"}; 
#' if \code{split.prob="exponential"}, the probability of splitting a node at depth \eqn{d} is \code{base}\eqn{^d}. 
#' @param split.prob A string indicating what kind of splitting probability is used for the tree prior. If 
#' \code{split.prob="polynomial"}, the splitting probability in Chipman et al. (2010) is used; 
#' If \code{split.prob="exponential"}, the splitting probability in Rockova and Saha (2019) is used.
#' @param ntree The number of trees in the ensemble.
#' @param ndpost The number of posterior samples returned.
#' @param nskip The number of posterior samples burned in.
#' @param keepevery Every \code{keepevery} posterior sample is kept to be returned to the user.
#' @param printevery As the MCMC runs, a message is printed every \code{printevery} iterations.
#' @param verbose A Boolean argument indicating whether any messages are printed out.
#' @param mc.cores The number of cores to employ in parallel.
#' @param nice Set the job niceness. The default niceness is \eqn{19} and niceness goes from \eqn{0} (highest) to \eqn{19} 
#' (lowest).
#' @param seed Seed required for reproducible MCMC.
#' 
#' @return The function \code{mc.backward.vs()} returns a list with the following components.
#' \item{best.model.names}{The vector of column names of the predictors selected by the backward selection approach.}
#' \item{best.model.cols}{The vector of column indices of the predictors selected by the backward selection approach.}
#' \item{best.model.order}{The step where the best model is located.}
#' \item{models}{The list of winner models from each step of the backward selection procedure; length equals \code{ncol{x}}.}
#' \item{model.errors}{The vector of MSEs (or MLLs if the response variable is binary) for the \code{ncol{x}} winner models.}
#' \item{elpd.loos}{The vector of LOO scores for the \code{ncol{x}} winner models.}
#' \item{all.models}{The list of all the evaluated models.}
#' \item{all.model.errors}{The vector of MSEs (or MLLs if the response variable is binary) for all the evaluated models.}
#' \item{precision}{The precision score for the backward selection approach; only returned if \code{true.idx} is provided.}
#' \item{recall}{The recall score for the backward selection approach; only returned if \code{true.idx} is provided.}
#' \item{f1}{The F1 score for the backward selection approach; only returned if \code{true.idx} is provided.}
#' \item{all.models.idx}{The vector of Boolean arguments indicating whether the corresponding model in \code{all.models} is
#' acceptable or not; a model containing all the relevant predictors is an acceptable model; only returned if \code{true.idx} 
#' is provided.}
#' 
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). 
#'   "BART: Bayesian additive regression trees."
#'    \emph{Ann. Appl. Stat.} \strong{4} 266--298.
#' 
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#'   
#' Rockova V, Saha E (2019). 
#'   “On theory for BART.” 
#'   \emph{In The 22nd International Conference on Artificial Intelligence and Statistics} (pp. 2839–2848). PMLR.
#'   
#' Vehtari, Aki, Andrew Gelman, and Jonah Gabry (2017).
#'   "Erratum to: Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC."
#'   \emph{Stat. Comput.} 27.5, p. 1433.
#' @seealso 
#' \code{\link{permute.vs}}, \code{\link{medianInclusion.vs}} and \code{\link{abc.vs}}.
#' @examples 
#' ## simulate data (Scenario C.C.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = friedman(100, 5, 1, FALSE)
#' ## parallel::mcparallel/mccollect do not exist on windows
#' if(.Platform$OS.type=='unix') {
#' ## test mc.backward.vs() function
#'   res = mc.backward.vs(data$X, data$Y, split.ratio=0.8, probit=FALSE, 
#'   true.idx=c(1:5), ntree=10, ndpost=100, nskip=100, mc.cores=2)
#' }
mc.backward.vs = function(x, 
                          y, 
                          split.ratio=0.8,
                          probit=FALSE, 
                          true.idx=NULL,
                          xinfo=matrix(0.0,0,0), 
                          numcut=100L,
                          usequants=FALSE, 
                          cont=FALSE, 
                          rm.const=TRUE, 
                          k=2.0, 
                          power=2.0, 
                          base=0.95,
                          split.prob="polynomial",
                          ntree=50L, 
                          ndpost=1000, 
                          nskip=1000,
                          keepevery=1L,
                          printevery=100L, 
                          verbose=FALSE,
                          mc.cores=2L, 
                          nice=19L, 
                          seed=99L) {
  
  #------------------------------
  # timer starts
  start = Sys.time()
  
  #------------------------------
  # parallel setting
  if(.Platform$OS.type!='unix')
    stop('parallel::mcparallel/mccollect do not exist on windows')
  
  RNGkind("L'Ecuyer-CMRG")
  set.seed(seed)
  parallel::mc.reset.stream()
  
  mc.cores.detected = detectCores()
  if(mc.cores > mc.cores.detected) mc.cores = mc.cores.detected
  
  #------------------------------
  # data
  n = nrow(x)
  p = ncol(x)

  ## split data into training and test
  train.idx = sample(1:n, size = floor(split.ratio * n), replace = FALSE)
  x.train = x[train.idx, ]
  y.train = y[train.idx]
  n.train = length(y.train)
  x.test = x[-train.idx, ]
  y.test = y[-train.idx]
  n.test = length(y.test)

  #------------------------------
  # returns
  models = list()        # winner models from each iteration of backward, length = p
  model.errors = c()     # winner models' model errors, length = p
  elpd.loos = c()        # Bayesian loo of winner models from each iteration of backward, length = p
  badKs = c()            # badKs from Bayesian loo for winner models from each iteration of backward, length = p
  
  all.models = list()    # all evaluated models
  all.model.errors = c() # model errors corresponding to all.models (same order)

  #------------------------------
  # model 0: full model
  models[[1]] = 1:p
  all.models[[1]] = 1:p
  temp.result = single.model.evaluation(x.train = x.train, y.train = y.train, x.test = x.test, y.test = y.test, 
                                        n.train = n.train, n.test = n.test, n = n, probit = probit,
                                        xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, 
                                        rm.const = rm.const, 
                                        k = k, power = power, base = base, split.prob = split.prob,
                                        ntree = ntree, ndpost = ndpost, nskip = nskip, 
                                        keepevery = keepevery, printevery = printevery, verbose = verbose)
  model.errors[1] = temp.result$model.error
  all.model.errors[1] = temp.result$model.error
  elpd.loos[1] = temp.result$elpd.loo
  badKs[1] = temp.result$badK
  
  rm(temp.result)

  #------------------------------
  # backward in sequential, parallel within each iteration
  for (i in 1:(p-1)) {
    ## model[[i]]: winner model from last iteration
    if(verbose) cat("iteration =", (i+1), "/")
    
    num.models = length(models[[i]])
    reduced.models = matrix(NA, nrow = num.models, ncol = (num.models - 1))   # each row is a new model
    for (j in 1:num.models) {
      reduced.models[j, ] = models[[i]][-j]
    }
    
    ## parallel
    if (num.models < mc.cores) {
      mc.cores = num.models
      mc.num.models = 1
    } else {
      mc.num.models = floor(num.models / mc.cores)
    }
    
    for(j in 1:mc.cores) {
      if (j <= num.models %% mc.cores){
        start.row = (j - 1) * (mc.num.models + 1) + 1
        end.row = j * (mc.num.models + 1)
      }
      else{
        start.row = (j - 1) * mc.num.models + 1 + num.models %% mc.cores
        end.row = j * mc.num.models + num.models %% mc.cores
      }
      
      
      parallel::mcparallel({psnice(value = nice);
        multiple.models.evaluation(x.train = x.train, y.train = y.train, x.test = x.test, y.test = y.test, 
                                   models = reduced.models[start.row:end.row, , drop=FALSE],
                                   n.train = n.train, n.test = n.test, n = n, probit = probit,
                                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, 
                                   rm.const = rm.const, 
                                   k = k, power = power, base = base, split.prob = split.prob,
                                   ntree = ntree, ndpost = ndpost, nskip = nskip, 
                                   keepevery = keepevery, printevery = printevery, verbose = verbose)},
        silent = (j != 1))
    }
    
    reduced.results.list = parallel::mccollect()
    
    ## collect parallel results
    reduced.results = reduced.results.list[[1]]$res
    reduced.models = reduced.results.list[[1]]$models
    
    if(mc.cores > 1) {
      for (j in 2:mc.cores) {
        reduced.results = rbind(reduced.results, reduced.results.list[[j]]$res)
        reduced.models = rbind(reduced.models, reduced.results.list[[j]]$models)
      }
    }
    
    rm(reduced.results.list)
    
    ## pick the winner model at this iteration
    l.star = which.min(reduced.results[, 1])
    models[[i+1]] = reduced.models[l.star, ]
    model.errors[i+1] = reduced.results[l.star, 1]
    elpd.loos[i+1] = reduced.results[l.star, 2]
    badKs[i+1] = reduced.results[l.star, 3]

    ## store results of all evaluated models
    for (j in 1:num.models) {
      all.models = c(all.models, list(reduced.models[j, ]))
    }
    all.model.errors = c(all.model.errors, reduced.results[, 1])
    
    rm(reduced.results)
  }
  
  if(verbose) cat("\n")
  
  #------------------------------
  # pick the best model
  best.model.order = which.max(elpd.loos)
  best.model.cols = models[[best.model.order]]
  best.model.names = dimnames(x)[[2]][best.model.cols]
  
  #------------------------------
  # returns
  res = list()
  res$best.model.names = best.model.names
  res$best.model.cols = best.model.cols
  res$best.model.order = best.model.order
  res$models = models
  res$model.errors = model.errors
  res$elpd.loos = elpd.loos
  res$badKs = badKs
  res$all.models = all.models
  res$all.model.errors = all.model.errors

  #------------------------------
  # score the results
  if(length(true.idx) > 0) {
    true.len = length(true.idx)
    
    tp = length(which(best.model.cols %in% true.idx))
    positive.len = length(best.model.cols)
    
    res$precision = (tp * 1.0) / (positive.len * 1.0)
    res$recall = (tp * 1.0) / (true.len * 1.0)
    res$f1 = 2 * res$precision * res$recall / (res$precision + res$recall)
  }
  
  #------------------------------
  # distinguish acceptable and unacceptable models
  if(length(true.idx) > 0) {
    all.models.idx = c()    # true if acceptable; false if unacceptable
    for (i in 1:length(all.models)) {
      all.models.idx[i] = all(true.idx %in% all.models[[i]])
    }
    res$all.models.idx = all.models.idx
  }
  
  #------------------------------
  # timer
  end = Sys.time()
  if(verbose) cat("Elapsed", end-start, '\n')
  
  return(res)
}

