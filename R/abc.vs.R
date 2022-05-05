#' Variable selection with ABC Bayesian forest
#' 
#' This function implements the variable selection approach proposed in Liu, Rockova and Wang (2021). Rockova and Pas (2020)
#' introduce a spike-and-forest prior which wraps the BART prior with a spike-and-slab prior on the model space. Due to intractable
#' marginal likelihood, Liu, Rockova and Wang (2021) propose an approximate Bayesian computation (ABC) sampling method based on 
#' data-splitting to help sample from the model space with higher ABC acceptance rate.
#' 
#' At each iteration of the algorithm, the data set is randomly split into a training set and a testing set according to a certain
#' split ratio. The algorithm proceeds by sampling a subset from the spike-and-slab prior on the model space, fitting a BART model
#' on the training set only with the predictors in the subset, and computing the root mean squared errors (RMSE) for the test set
#' based on a posterior sample from the fitted BART model. Only those subsets that result in a low RMSE on the test set are kept 
#' for selection. ABC Bayesian forest selects predictors based on their marginal posterior variable inclusion probabilities (MPVIPs)
#' which are estimated by computing the proportion of ABC accepted BART posterior samples that use the predictor at least one time.
#' Given the MPVIPs, predictors with MPVIP exceeding a pre-specified threshold are selected.\cr
#' See Liu, Rockova and Wang (2021) or Section 2.2.4 in Luo and Daniels (2021) for details.
#' 
#' @param x A matrix or a data frame of predictors values with each row corresponding to an observation and each column 
#' corresponding to a predictor. If a predictor is a factor with \eqn{q} levels in a data frame, it is replaced with \eqn{q} dummy 
#' variables.
#' @param y A vector of response (continuous or binary) values.
#' @param nabc The number of ABC samples, i.e., the number of subsets sampled from the model space.
#' @param tolerance A number between \eqn{0} and \eqn{1}; the \code{nabc} subsets are ranked by MSE in ascending order if the 
#' response variable is continuous (or by mean log loss (MLL) if the response variable is binary), and the top \code{tolerance*100}\%
#' of the subsets are accepted by ABC for selection.
#' @param threshold A number between \eqn{0} and \eqn{1}; within the ABC accepted subsets, predictors with MPVIP exceeding 
#' \code{threshold} are selected.
#' @param beta.params A vector with two positive numbers; the spike-and-slab prior on the model space is assumed to be a beta-binomial
#' prior, i.e., \eqn{\theta ~ }Beta(\code{beta.params[1], beta.params[2]}) and each predictor is included into a model by 
#' Bernoulli(\eqn{\theta}); only used when \code{beta.theta=NA}.
#' @param beta.theta A number between \eqn{0} and \eqn{1}; the probability that a predictor is included into a model; 
#' if \code{beta.theta=NA}, it is sampled from Beta(\code{beta.params[1], beta.params[2]}).
#' @param split.ratio A number between \eqn{0} and \eqn{1}; the data set \code{(x, y)} is split into a training set and a testing
#' set according to the \code{split.ratio}.
#' @param probit A Boolean argument indicating whether the response variable is binary or continuous; \code{probit=FALSE} (by default)
#' means that the response variable is continuous. 
#' @param true.idx (Optional) A vector of indices of the true relevant predictors; if \code{true.idx} is provided and 
#' \code{analysis=TRUE}, metrics including precision, recall and F1 score are returned.
#' @param analysis A Boolean argument indicating whether to perform variable selection; if \code{analysis=TRUE}, the best model
#' selected by ABC Bayesian forest is returned; if \code{analysis=FALSE}, only return the visited models and their corresponding model
#' errors.
#' @param sparse A Boolean argument indicating whether to perform DART or BART.
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
#' 
#' @return The function \code{abc.vs()} returns a list with the following components.
#' \item{theta}{The probability that a predictor is included into a model.}
#' \item{models}{A matrix with \code{nabc} rows and \code{ncol(x)} columns; each row corresponds to a ABC model (or subset); if the
#' (\eqn{i, j})-th element is \eqn{1}, it means that the \eqn{j}-th predictor is included in the \eqn{i}-th ABC model; if the
#' (\eqn{i, j})-th element is \eqn{0}, it means that the \eqn{j}-th predictor is not included in the \eqn{i}-th ABC model.}
#' \item{actual.models}{A matrix with \code{nabc} rows and \code{ncol(x)} columns; each row corresponds to a ABC BART posterior 
#' sample; if the (\eqn{i, j})-th element is \eqn{1}, it means that the \eqn{j}-th predictor is used as a split variable at least 
#' one time in the BART posterior sample of the \eqn{i}-th ABC model; if the (\eqn{i, j})-th element is \eqn{0}, it means that the 
#' \eqn{j}-th predictor is not used as a split variable in the BART posterior sample of the \eqn{i}-th ABC model.}
#' \item{model.errors}{The vector of MSEs (or MLLs if the response variable is binary) for the \code{nabc} ABC models.}
#' \item{idx}{The vector of indices (in terms of the row numbers of \code{models}) of the ABC accepted models which are the top 
#' \code{tolerance*100}\% of the \code{nabc} ABC models when ranked by MSE or MLL in ascending order; only returned when
#' \code{analysis=TRUE}.}
#' \item{top.models}{A matrix with \code{length(idx)} rows and \code{ncol(x)} columns, representing the ABC accepted models;
#' \code{top.models=models[idx, ]}; only returned when \code{analysis=TRUE}.}
#' \item{top.actual.models}{A matrix with \code{length(idx)} rows and \code{ncol(x)} columns, representing the ABC accepted BART
#' posterior samples; \code{top.models=actual.models[idx, ]}; only returned when \code{analysis=TRUE}.}
#' \item{mip}{The vector of marginal posterior variable inclusion probabilities; only returned when \code{analysis=TRUE}.}
#' \item{best.model}{The vector of predictors selected by ABC Bayesian forest; only returned when \code{analysis=TRUE}.}
#' \item{precision}{The precision score for the ABC Bayesian forest; only returned when \code{analysis=TRUE} and \code{true.idx} is 
#' provided.}
#' \item{recall}{The recall score for the ABC Bayesian forest; only returned when \code{analysis=TRUE} and \code{true.idx} is 
#' provided.}
#' \item{f1}{The F1 score for the ABC Bayesian forest; only returned when \code{analysis=TRUE} and \code{true.idx} is provided.}
#' 
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references
#' Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). 
#'   "BART: Bayesian additive regression trees."
#'    \emph{Ann. Appl. Stat.} \strong{4} 266--298.
#'    
#' Linero, A. R. (2018). 
#'   "Bayesian regression trees for high-dimensional prediction and variable selection." 
#'   \emph{J. Amer. Statist. Assoc.} \strong{113} 626--636.
#'   
#' Liu, Yi, Veronika Rockova, and Yuexi Wang (2021).
#'   "Variable selection with ABC Bayesian forests."
#'   \emph{J. R. Stat. Soc. Ser. B. Stat. Methodol.} 83.3, pp. 453--481.
#'   
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#'   
#' Rockova Veronika and Stephanie van der Pas (2020).
#'   "Posterior concentration for Bayesian regression trees and forests."
#'   \emph{Ann. Statist.} 48.4, pp. 2108--2131.
#' @seealso 
#' \code{\link{permute.vs}}, \code{\link{medianInclusion.vs}} and \code{\link{mc.backward.vs}}.
#' @examples 
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## test abc.vs() function
#' res = abc.vs(data$X, data$Y, nabc=100, tolerance=0.1, threshold=0.25, beta.params=c(1.0, 1.0), 
#' split.ratio=0.5, probit=FALSE, true.idx=c(1,2,6:8), ntree=10, ndpost=1, nskip=200, analysis=TRUE)
abc.vs = function(x, 
                  y, 
                  nabc=1000, 
                  tolerance=0.1, 
                  threshold=0.25,
                  beta.params=c(1.0, 1.0), 
                  beta.theta=NA,
                  split.ratio=0.5, 
                  probit=FALSE, 
                  true.idx=NULL,
                  analysis=TRUE,
                  sparse=FALSE,
                  xinfo=matrix(0.0,0,0), 
                  numcut=100L,
                  usequants=FALSE, 
                  cont=FALSE, 
                  rm.const=TRUE, 
                  k=2.0, 
                  power=2.0, 
                  base=0.95,
                  split.prob="polynomial",
                  ntree=10L, 
                  ndpost=1, 
                  nskip=200,
                  keepevery=1L, 
                  printevery=100L, 
                  verbose=FALSE) {
  
  res = list()
  
  #------------------------------
  # timer starts
  #start = Sys.time()
  
  #------------------------------
  # data
  n = nrow(x)
  p = ncol(x)
  
  #------------------------------
  # returns
  models = matrix(NA, nrow = 1, ncol = p)  ## pool of available predictors
  actual.models = matrix(NA, nrow = 1, ncol = p)  ## predictors used in the BART ensemble
  model.errors = c()  ## MSE for continuous y and mean log loss for binary y
  
  #------------------------------
  # theta ~ Beta(betaparams)
  if(is.na(beta.theta)) beta.theta = rbeta(1, beta.params[1], beta.params[2])
  res$theta = beta.theta
  
  #------------------------------
  # run ABC Bayesian forest
  for (i in 1:nabc) {
    ##---------------------
    ## split data
    train.idx = sample(1:n, size = floor(split.ratio * n), replace = FALSE)
    x.train = x[train.idx, ]
    y.train = y[train.idx]
    n.train = length(y.train)
    x.test = x[-train.idx, ]
    y.test = y[-train.idx]
    n.test = length(y.test)
    
    ##---------------------
    ## pick a subset
    current.model = c()
    
    ## make sure the model selected contains at least one predictor
    while (!any(current.model == 1)) {
      current.model = c()
      
      for (j in 1:p) {
        if(rbinom(1, 1, beta.theta))
          current.model[j] = 1
        else
          current.model[j] = 0
      }
    }
    
    if(i == 1)
      models[1, ] = current.model
    else
      models = rbind(models, current.model)
    
    ##---------------------
    ## bart
    if(probit) {
      bart.obj = pbart(x.train = x.train[, which(current.model == 1)], y.train = y.train, 
                       x.test = x.test[, which(current.model == 1)], sparse = sparse,
                       xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                       k = k, power = power, base = base, split.prob = split.prob,
                       ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                       printevery = printevery, verbose = verbose)
      
      ## model error: mean logarithmic loss
      prob.test = bart.obj$prob.test
      model.error = 0
      for (j in 1:n.test) {
        if(((prob.test[j] == 0) & (y.test[j] == 0)) | ((prob.test[j] == 1) & (y.test[j] == 1))){
          model.error = model.error
        } else {
          model.error = model.error + y.test[j] * log(prob.test[j]) + (1 - y.test[j]) * log(1 - prob.test[j])
        }
      }
      model.errors[i] = - model.error / n.test
      
      actual.current.model = rep(0, p)
      actual.current.model[which(current.model == 1)[which(bart.obj$varcount > 0)]] = 1
      
      if(i == 1)
        actual.models[1, ] = actual.current.model
      else
        actual.models = rbind(actual.models, actual.current.model)
      
    }else{
      bart.obj = wbart(x.train = x.train[, which(current.model == 1)], y.train = y.train, 
                       x.test = x.test[, which(current.model == 1)], sparse = sparse,
                       xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                       k = k, power = power, base = base, split.prob = split.prob,
                       ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                       printevery = printevery, verbose = verbose)
      
      pseudo.y.test = bart.obj$yhat.test + rnorm(n.test, 0, bart.obj$sigma[nskip + ndpost])
      model.errors[i] = sqrt(mean((pseudo.y.test - y.test) ^ 2))
      
      actual.current.model = rep(0, p)
      actual.current.model[which(current.model==1)[which(bart.obj$varcount > 0)]] = 1
      
      if(i == 1)
        actual.models[1, ] = actual.current.model
      else
        actual.models = rbind(actual.models, actual.current.model)
    }
    
  }
  
  if(analysis) {
    #------------------------------
    # top models
    ntop = ceiling(nabc * tolerance)
    idx0 = sort(model.errors, index.return = TRUE)$ix
    idx = idx0[1 : ntop]
    
    top.models = models[idx, ]
    top.actual.models = actual.models[idx, ]
    
    res$models = models
    res$actual.models = actual.models
    res$model.errors = model.errors
    res$idx = idx
    res$top.models = top.models
    res$top.actual.models = top.actual.models
    
    #------------------------------
    # marginal inclusion probabilities
    mip = colMeans(top.actual.models)
    best.model = which(mip >= threshold)
    
    res$mip = mip
    res$best.model = best.model
    
    #------------------------------
    # score
    if(length(true.idx) > 0) {
      true.len = length(true.idx)
      
      tp = length(which(best.model %in% true.idx))
      positive.len = length(best.model)
      
      res$precision = (tp * 1.0) / (positive.len * 1.0)
      res$recall = (tp * 1.0) / (true.len * 1.0)
      res$f1 = 2 * res$precision * res$recall / (res$precision + res$recall)
    }
  }else{
    res$models = models
    res$actual.models = actual.models
    res$model.errors = model.errors
  }
 
  
  #------------------------------
  # timer ends
  #end = Sys.time()
  #cat("Elapsed", end-start, '\n')
  
  return(res)
}