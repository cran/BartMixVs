#' Variable selection with ABC Bayesian forest (using parallel computation)
#' 
#' This function implements the variable selection approach proposed in Liu, Rockova and Wang (2021) with parallel computation.
#' Rockova and Pas (2020) introduce a spike-and-forest prior which wraps the BART prior with a spike-and-slab prior on the model 
#' space. Due to intractable marginal likelihood, Liu, Rockova and Wang (2021) propose an approximate Bayesian computation (ABC) 
#' sampling method based on data-splitting to help sample from the model space with higher ABC acceptance rate.\cr
#' Unlike the function \code{abc.vs()} which sequentially evaluates ABC models, this function evaluates the ABC models in parallel.
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
#' Bernoulli(\eqn{\theta}).
#' @param split.ratio A number between \eqn{0} and \eqn{1}; the data set \code{(x, y)} is split into a training set and a testing
#' set according to the \code{split.ratio}.
#' @param probit A Boolean argument indicating whether the response variable is binary or continuous; \code{probit=FALSE} (by default)
#' means that the response variable is continuous. 
#' @param true.idx (Optional) A vector of indices of the true relevant predictors; if \code{true.idx} is provided, metrics including 
#' precision, recall and F1 score are returned.
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
#' @param mc.cores The number of cores to employ in parallel.
#' @param nice Set the job niceness. The default niceness is \eqn{19} and niceness goes from \eqn{0} (highest) to \eqn{19} 
#' (lowest).
#' @param seed Seed required for reproducible MCMC.
#' 
#' @return The function \code{mc.abc.vs()} returns a list with the following components.
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
#' \code{tolerance*100}\% of the \code{nabc} ABC models when ranked by MSE or MLL in ascending order.}
#' \item{top.models}{A matrix with \code{length(idx)} rows and \code{ncol(x)} columns, representing the ABC accepted models;
#' \code{top.models=models[idx, ]}.}
#' \item{top.actual.models}{A matrix with \code{length(idx)} rows and \code{ncol(x)} columns, representing the ABC accepted BART
#' posterior samples; \code{top.models=actual.models[idx, ]}.}
#' \item{mip}{The vector of marginal posterior variable inclusion probabilities.}
#' \item{best.model}{The vector of predictors selected by ABC Bayesian forest.}
#' \item{precision}{The precision score for the ABC Bayesian forest; only returned when \code{true.idx} is provided.}
#' \item{recall}{The recall score for the ABC Bayesian forest; only returned when \code{true.idx} is provided.}
#' \item{f1}{The F1 score for the ABC Bayesian forest; only returned when \code{true.idx} is provided.}
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
#' \code{\link{abc.vs}}.
#' @examples 
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## parallel::mcparallel/mccollect do not exist on windows
#' if(.Platform$OS.type=='unix') {
#' ## test mc.abc.vs() function
#'   res = mc.abc.vs(data$X, data$Y, nabc=100, tolerance=0.1, threshold=0.25, 
#'   beta.params=c(1.0, 1.0), split.ratio=0.5, probit=FALSE, true.idx=c(1,2,6:8), 
#'   ntree=10, ndpost=1, nskip=200, mc.cores=2)
#' }
mc.abc.vs = function(x, 
                     y, 
                     nabc=1000, 
                     tolerance=0.1, 
                     threshold=0.25,
                     beta.params=c(1.0, 1.0),
                     split.ratio=0.5, 
                     probit=FALSE, 
                     true.idx=NULL,
                     sparse = FALSE,
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
                     verbose=FALSE,
                     mc.cores = 2L, 
                     nice = 19L, 
                     seed = 99L) {
  
  #------------------------------
  # timer starts
  start = Sys.time()

  res = list()
  
  #------------------------------
  # parallel setting
  if(.Platform$OS.type!='unix')
    stop('parallel::mcparallel/mccollect do not exist on windows')
  
  RNGkind("L'Ecuyer-CMRG")
  set.seed(seed)
  parallel::mc.reset.stream()
  
  mc.cores.detected = detectCores()
  if(mc.cores > mc.cores.detected) mc.cores = mc.cores.detected
  if(nabc < mc.cores) mc.cores = nabc
  
  #------------------------------
  # parallel running
  beta.theta = rbeta(1, beta.params[1], beta.params[2])
  res$theta = beta.theta
  
  for(j in 1:mc.cores) {
    
    if (j <= nabc %% mc.cores){
      mc.nabc = floor(nabc / mc.cores) + 1
    }else{
      mc.nabc = floor(nabc / mc.cores)
    }
    
    parallel::mcparallel({psnice(value = nice);
      abc.vs(x = x, y = y, nabc = mc.nabc, tolerance = tolerance, threshold = threshold,
             beta.params = beta.params, beta.theta = beta.theta, split.ratio = split.ratio, 
             probit = probit, true.idx = true.idx, analysis = FALSE, sparse = sparse,
             xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, 
             rm.const = rm.const, k = k, power = power, base = base, split.prob = split.prob,
             ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
             printevery = printevery, verbose = verbose)},
      silent = (j != 1))
  }
  
  results.list = parallel::mccollect()
  
  # collect parallel results
  models = results.list[[1]]$models
  actual.models = results.list[[1]]$actual.models
  model.errors = results.list[[1]]$model.errors
  
  if(mc.cores > 1) {
    for (j in 2:mc.cores) {
      models = rbind(models, results.list[[j]]$models)
      actual.models = rbind(actual.models, results.list[[j]]$actual.models)
      model.errors = c(model.errors, results.list[[j]]$model.errors)
    }
  }
  
  res$models = models
  res$actual.models = actual.models
  res$model.errors = model.errors
  
  rm(results.list)
  
  
  #------------------------------
  # top models
  ntop = ceiling(nabc * tolerance)
  idx0 = sort(model.errors, index.return = TRUE)$ix
  idx = idx0[1 : ntop]
  
  top.models = models[idx, ]
  top.actual.models = actual.models[idx, ]
  
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
  
  #------------------------------
  # timer
  end = Sys.time()
  if(verbose) cat("Elapsed", end-start, '\n')
  
  return(res)
}