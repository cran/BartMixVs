multiple.bart = function(x.train, 
                         y.train, 
                         probit=TRUE, 
                         categorical.idx=NULL,
                         xinfo=matrix(0.0,0,0), 
                         numcut=100L, 
                         usequants=FALSE, 
                         cont=FALSE, 
                         rm.const=TRUE, 
                         k=2.0, 
                         power=2.0, 
                         base=0.95,
                         split.prob="polynomial",
                         ntree=20L, 
                         ndpost=1000, 
                         nskip=1000, 
                         keepevery=1L, 
                         verbose=FALSE) {
  
  #------------------------
  # data
  if (is.vector(y.train)) npermute = 1
  else npermute = ncol(y.train)
  
  #------------------------
  # set up permute matrix
  permute.vips = matrix(NA, nrow = npermute, ncol = ncol(x.train))
  permute.mis = matrix(NA, nrow = npermute, ncol = ncol(x.train))
  if (length(categorical.idx) > 0)
    permute.within.type.vips = matrix(NA, nrow = npermute, ncol = ncol(x.train))
  
  #------------------------
  # run multiple bart models
  if (npermute == 1) {
    if (probit) {
      bart = pbart(x.train = x.train, y.train = y.train, sparse = FALSE,
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                   k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
    } else {
      bart = wbart(x.train = x.train, y.train = y.train, sparse = FALSE,
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                   k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
    }
    
    permute.vips[1, ] = bart$vip
    permute.mis[1, ] = bart$mi
    if (length(categorical.idx) > 0)
      permute.within.type.vips[1, ] = bart$within.type.vip
    
  } else {
    for (i in 1:npermute) {
      if (probit) {
        bart = pbart(x.train = x.train, y.train = y.train[, i], sparse = FALSE,
                     xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                     k = k, power = power, base = base, split.prob = split.prob,
                     ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
      } else {
        bart = wbart(x.train = x.train, y.train = y.train[, i], sparse = FALSE,
                     xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                     k = k, power = power, base = base, split.prob = split.prob,
                     ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
      }
      
      permute.vips[i, ] = bart$vip
      permute.mis[i, ] = bart$mi
      if (length(categorical.idx) > 0)
        permute.within.type.vips[i, ] = bart$within.type.vip
    }
  }
  
  res = list()
  res$permute.vips = permute.vips
  res$permute.mis = permute.mis
  if (length(categorical.idx) > 0)
    res$permute.within.type.vips = permute.within.type.vips
  
  return(res)
}

#' Permutation-based variable selection approach with parallel computation
#' 
#' This function implements the permutation-based variable selection approach for BART (see Algorithm 1 in Luo and Daniels (2021)
#' for details) with parallel computation used in computing the null variable importance scores. Three types of variable importance 
#' measures are considered: BART variable inclusion proportions (VIP), BART within-type variable inclusion proportions (within-type 
#' VIP) and BART Metropolis Importance (MI).\cr
#' The permutation-based variable selection approach using BART VIP as the variable importance measure is proposed by Bleich et al.
#' (2014). BART within-type VIP and BART MI are proposed by Luo and Daniels (2021), for the sake of the existence of mixed-type 
#' predictors and the goal of allowing more relevant predictors into the model.
#' 
#' The detailed algorithm can be found in Algorithm 1 in Luo and Daniels (2021). The permutation-based variable selection approach
#' using within-type VIP as the variable importance measure is only used when the predictors are of mixed-type; otherwise, it is
#' the same as the one using VIP as the variable importance measure.\cr
#' If \code{true.idx} is provided, the precision, recall and F1 scores will be returned for the three (or two if the predictors are
#' of the same type) methods.\cr
#' If \code{plot=TRUE}, three (or two if the predictors are of the same type) plots showing which predictors are selected 
#' are generated.
#' 
#' @param x.train A matrix or a data frame of predictors values with each row corresponding to an observation and each column 
#' corresponding to a predictor. If a predictor is a factor with \eqn{q} levels in a data frame, it is replaced with \eqn{q} dummy 
#' variables.
#' @param y.train A vector of response (continuous or binary) values.
#' @param probit A Boolean argument indicating whether the response variable is binary or continuous; \code{probit=FALSE} (by default)
#' means that the response variable is continuous. 
#' @param npermute The number of permutations for estimating the null distributions of the variable importance scores.
#' @param nreps The number of replications for obtaining the averaged (or median) variable importance scores based on the original 
#' data set.
#' @param alpha A number between \eqn{0} and \eqn{1}; a predictor is selected if its averaged (or median) variable importance score 
#' exceeds the \eqn{1-\alpha} quantile of the corresponding null distribution.
#' @param true.idx (Optional) A vector of indices of the true relevant predictors; if provided, metrics including precision, recall
#' and F1 score will be returned.
#' @param plot (Optional) A Boolean argument indicating whether plots are returned or not.
#' @param n.var.plot The number of variables to be plotted.
#' @param xinfo A matrix of cut-points with each row corresponding to a predictor and each column corresponding to a cut-point.
#' \code{xinfo=matrix(0.0,0,0)} indicates the cut-points are specified by BART.
#' @param numcut The number of possible cut-points; If a single number is given, this is used for all predictors; 
#' Otherwise a vector with length equal to \code{ncol(x.train)} is required, where the \eqn{i-}th element gives the number of 
#' cut-points for the \eqn{i-}th predictor in \code{x.train}. If \code{usequants=FALSE}, \code{numcut} equally spaced 
#' cut-points are used to cover the range of values in the corresponding column of \code{x.train}. 
#' If \code{usequants=TRUE}, then min(\code{numcut}, the number of unique values in the corresponding column of 
#' \code{x.train} - 1) cut-point values are used.
#' @param usequants A Boolean argument indicating how the cut-points in \code{xinfo} are generated; 
#' If \code{usequants=TRUE}, uniform quantiles are used for the cut-points; Otherwise, the cut-points are generated uniformly.
#' @param cont A Boolean argument indicating whether to assume all predictors are continuous.
#' @param rm.const A Boolean argument indicating whether to remove constant predictors.
#' @param k The number of prior standard deviations that \eqn{E(Y|x) = f(x)} is away from \eqn{+/-.5}. The response 
#' (\code{y.train}) is internally scaled to the range from \eqn{-.5} to \eqn{.5}. The bigger \code{k} is, the more conservative 
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
#' @param verbose A Boolean argument indicating whether any messages are printed out.
#' @param mc.cores The number of cores to employ in parallel.
#' @param nice Set the job niceness. The default niceness is \eqn{19} and niceness goes from \eqn{0} (highest) to \eqn{19} 
#' (lowest).
#' @param seed Seed required for reproducible MCMC.
#' 
#' @return The function \code{mc.permute.vs()} returns three (or two if the predictors are of the same type) plots 
#' if \code{plot=TRUE} and a list with the following components.
#' \item{vip.imp.cols}{The vector of column indices of the predictors selected by the approach using VIP as the variable importance
#' score.}
#' \item{vip.imp.names}{The vector of column names of the predictors selected by the approach using VIP as the variable importance
#' score.}
#' \item{avg.vip}{The vector (length=\code{ncol(x.train)}) of the averaged VIPs based on the original data set; 
#' \code{avg.vip=colMeans(avg.vip.mtx)}.}
#' \item{avg.vip.mtx}{A matrix of VIPs based on the original data set, with each row corresponding to a repetition and each column
#' corresponding to a predictor.}
#' \item{permute.vips}{A matrix of VIPs based on the null data sets, with each row corresponding to a permutation (null data set) 
#' and each column corresponding to a predictor.}
#' \item{within.type.vip.imp.cols}{The vector of column indices of the predictors selected by the approach using within-type VIP 
#' as the variable importance score.}
#' \item{within.type.vip.imp.names}{The vector of column names of the predictors selected by the approach using within-type VIP 
#' as the variable importance score.}
#' \item{avg.within.type.vip}{The vector (length=\code{ncol(x.train)}) of the averaged within-type VIPs based on the original data 
#' set; \code{avg.within.type.vip=colMeans(avg.within.type.vip.mtx)}.}
#' \item{avg.within.type.vip.mtx}{A matrix of within-type VIPs based on the original data set, with each row corresponding to a 
#' repetition and each column corresponding to a predictor.}
#' \item{permute.within.type.vips}{A matrix of within VIPs based on the null data sets, with each row corresponding to a permutation 
#' (null data set) and each column corresponding to a predictor.}
#' \item{mi.imp.cols}{The vector of column indices of the predictors selected by the approach using MI as the variable importance
#' score.}
#' \item{mi.imp.names}{The vector of column names of the predictors selected by the approach using MI as the variable importance
#' score.}
#' \item{median.mi}{The vector (length=\code{ncol(x.train)}) of the median MIs based on the original data set; 
#' \code{median.mi=colMeans(median.mi.mtx)}.}
#' \item{median.mi.mtx}{A matrix of MIs based on the original data set, with each row corresponding to a repetition and each column
#' corresponding to a predictor.}
#' \item{permute.mis}{A matrix of MIs based on the null data sets, with each row corresponding to a permutation (null data set) 
#' and each column corresponding to a predictor.}
#' \item{true.idx}{A vector of indices of the true relevant predictors; only returned if \code{true.idx} is provided as inputs.}
#' \item{vip.precision}{The precision score for the approach using VIP as the variable importance score; only returned if 
#' \code{true.idx} is provided.}
#' \item{vip.recall}{The recall score for the approach using VIP as the variable importance score; only returned if \code{true.idx} 
#' is provided.}
#' \item{vip.f1}{The F1 score for the approach using VIP as the variable importance score; only returned if \code{true.idx} 
#' is provided.}
#' \item{wt.vip.precision}{The precision score for the approach using within-VIP as the variable importance score; only returned when
#' the predictors are of the same type and \code{true.idx} is provided.}
#' \item{wt.vip.recall}{The recall score for the approach using within-VIP as the variable importance score; only returned when
#' the predictors are of the same type and \code{true.idx} is provided.}
#' \item{wt.vip.f1}{The F1 score for the approach using within-VIP as the variable importance score; only returned when
#' the predictors are of the same type and \code{true.idx} is provided.}
#' \item{mi.precision}{The precision score for the approach using MI as the variable importance score; only returned if \code{true.idx} 
#' is provided.}
#' \item{mi.recall}{The recall score for the approach using MI as the variable importance score; only returned if \code{true.idx} 
#' is provided.}
#' \item{mi.f1}{The F1 score for the approach using MI as the variable importance score; only returned if \code{true.idx} 
#' is provided.}
#' 
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Bleich, Justin et al. (2014).
#'   "Variable selection for BART: an application to gene regulation."
#'   \emph{Ann. Appl. Stat.} 8.3, pp 1750--1781.
#' 
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
#' @seealso 
#' \code{\link{permute.vs}}, \code{\link{medianInclusion.vs}}, \code{\link{mc.backward.vs}} and \code{\link{abc.vs}}.
#' @examples 
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## parallel::mcparallel/mccollect do not exist on windows
#' if(.Platform$OS.type=='unix') {
#' ## test mc.permute.vs() function
#'   res = mc.permute.vs(data$X, data$Y, probit=FALSE, npermute=100, nreps=10, alpha=0.05, 
#'   true.idx=c(1, 2, 6:8), plot=FALSE, ntree=10, ndpost=100, nskip=100, mc.cores=2)
#' }
mc.permute.vs = function(x.train, 
                         y.train,
                         probit=FALSE, 
                         npermute=100L, 
                         nreps=10L,                          ## number of replicates 
                         alpha=0.05,                            ## local threshold
                         true.idx=NULL, 
                         plot=TRUE, 
                         n.var.plot=Inf,
                         xinfo=matrix(0.0,0,0), 
                         numcut=100L,    ## xinfo parameters
                         usequants=FALSE, 
                         cont=FALSE, 
                         rm.const=TRUE, 
                         k=2.0, 
                         power=2.0, 
                         base=0.95,
                         split.prob="polynomial",
                         ntree=20L,
                         ndpost=1000, 
                         nskip=1000,
                         keepevery=1L, 
                         verbose=FALSE, 
                         mc.cores=2L, 
                         nice=19L,
                         seed=99L) {
  
  if(.Platform$OS.type!='unix')
    stop('parallel::mcparallel/mccollect do not exist on windows')
  
  RNGkind("L'Ecuyer-CMRG")
  set.seed(seed)
  parallel::mc.reset.stream()
  
  res = list()
  
  #------------------------------
  # timer starts
  start = Sys.time()

  
  #-----------------------------------------------------------
  # data
  categorical.idx = which(sapply(x.train, function(s) {is.factor(s)}))
  categorical.names = names(categorical.idx)
  
  #-----------------------------------------------------------
  # get avg/median variable importance from the original data
  if(verbose) cat("original data set...")
  
  avg.vip.mtx = matrix(NA, nrow = nreps, ncol = ncol(x.train))
  median.mi.mtx = matrix(NA, nrow = nreps, ncol = ncol(x.train))
  if (length(categorical.idx) > 0)
    avg.within.type.vip.mtx = matrix(NA, nrow = nreps, ncol = ncol(x.train))
  
  cnt = 0
  while (cnt < nreps) {
    if (probit) {
      bart = pbart(x.train = x.train, y.train = y.train, sparse = FALSE,
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                   k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
    } else {
      bart = wbart(x.train = x.train, y.train = y.train, sparse = FALSE,
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                   k = k, power = power, base = base, split.prob = split.prob,
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose) 
    }
    
    cnt = cnt + 1
    avg.vip.mtx[cnt, ] = bart$vip
    median.mi.mtx[cnt, ] = bart$mi
    if (length(categorical.idx) > 0)
      avg.within.type.vip.mtx[cnt, ] = bart$within.type.vip
  }
  
  avg.vip = colMeans(avg.vip.mtx)
  names(avg.vip) = colnames(x.train)
  avg.vip = sort(avg.vip, decreasing = T)
  
  median.mi = apply(median.mi.mtx, 2, median)
  names(median.mi) = colnames(x.train)
  median.mi = sort(median.mi, decreasing = T)
  
  if (length(categorical.idx) > 0) {
    avg.within.type.vip = colMeans(avg.within.type.vip.mtx)
    names(avg.within.type.vip) = colnames(x.train)
    avg.within.type.vip = sort(avg.within.type.vip, decreasing = T)
  }
  
  if(verbose) cat("complete! \n")
  
  
  #-----------------------------------------------------------
  # build null permutation
  if(verbose) cat('null data sets...')
  
  ## parallel
  mc.cores.detected = detectCores()
  if(mc.cores > mc.cores.detected) mc.cores = mc.cores.detected
  mc.npermute = floor(npermute / mc.cores)
  
  y.permuted = sapply(1:npermute, function(s) sample(y.train, size = length(y.train), replace = F))  # n x npermute
  
  for(i in 1:mc.cores) {
    if (i <= npermute %% mc.cores){
      start.col = (i - 1) * (mc.npermute + 1) + 1
      end.col = i * (mc.npermute + 1)
    }
    else{
      start.col = (i - 1) * mc.npermute + 1 + npermute %% mc.cores
      end.col = i * mc.npermute + npermute %% mc.cores
    }
    
    parallel::mcparallel({psnice(value = nice);
      multiple.bart(x.train = x.train, y.train = y.permuted[, start.col:end.col], 
                    probit = probit, categorical.idx = categorical.idx,
                    xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const,
                    k = k, power = power, base = base, split.prob = split.prob,
                    ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, verbose = verbose)},
      silent = (i != 1))
  }
  
  permute.list = parallel::mccollect()
  
  ## collect parallel results
  permute.vips = permute.list[[1]]$permute.vips
  permute.mis = permute.list[[1]]$permute.mis
  if (length(categorical.idx) > 0) permute.within.type.vips = permute.list[[1]]$permute.within.type.vips
  
  if(mc.cores > 1) {
    for (i in 2:mc.cores) {
      permute.vips = rbind(permute.vips, permute.list[[i]]$permute.vips)
      permute.mis = rbind(permute.mis, permute.list[[i]]$permute.mis)
      
      if (length(categorical.idx) > 0) 
        permute.within.type.vips = rbind(permute.within.type.vips, permute.list[[i]]$permute.within.type.vips)
    }
  }
  
  rm(permute.list)
  
  if(verbose) cat("complete! \n")
  
  #-----------------------------------------------------------
  # sort permute mat and return results
  colnames(permute.vips) = colnames(x.train)
  permute.vips = permute.vips[, names(avg.vip)]
  
  colnames(permute.mis) = colnames(x.train)
  permute.mis = permute.mis[, names(median.mi)]
  
  if (length(categorical.idx) > 0) {
    colnames(permute.within.type.vips) = colnames(x.train)
    permute.within.type.vips = permute.within.type.vips[, names(avg.within.type.vip)]
  }
  
  #-----------------------------------------------------------
  # use local cutoff & returns
  vip.pointwise.cutoffs = apply(permute.vips, 2, quantile, probs = 1 - alpha)
  vip.imp.names = names(avg.vip[(avg.vip > vip.pointwise.cutoffs) & (avg.vip > 0)])
  vip.imp.cols = sapply(1:length(vip.imp.names), function(x) {which(vip.imp.names[x] == colnames(x.train))})
  res$vip.imp.cols = vip.imp.cols
  res$vip.imp.names = vip.imp.names
  res$avg.vip = avg.vip
  res$avg.vip.mtx = avg.vip.mtx
  res$permute.vips = permute.vips
  
  mi.pointwise.cutoffs = apply(permute.mis, 2, quantile, probs = 1 - alpha)
  mi.imp.names = names(median.mi[(median.mi > mi.pointwise.cutoffs) & (median.mi > 0)])
  mi.imp.cols = sapply(1:length(mi.imp.names), function(x) {which(mi.imp.names[x] == colnames(x.train))})
  res$mi.imp.cols = mi.imp.cols
  res$mi.imp.names = mi.imp.names
  res$median.mi = median.mi
  res$median.mi.mtx = median.mi.mtx
  res$permute.mis = permute.mis
  
  if (length(categorical.idx) > 0) {
    within.type.vip.pointwise.cutoffs = apply(permute.within.type.vips, 2, quantile, probs = 1 - alpha)
    within.type.vip.imp.names = names(avg.within.type.vip[(avg.within.type.vip > within.type.vip.pointwise.cutoffs) 
                                                          & (avg.within.type.vip > 0)])
    within.type.vip.imp.cols = sapply(1:length(within.type.vip.imp.names), 
                                      function(x) {which(within.type.vip.imp.names[x] == colnames(x.train))})
    
    res$within.type.vip.imp.cols = within.type.vip.imp.cols
    res$within.type.vip.imp.names = within.type.vip.imp.names
    res$avg.within.type.vip = avg.within.type.vip
    res$avg.within.type.vip.mtx = avg.within.type.vip.mtx
    res$permute.within.type.vips = permute.within.type.vips
  }
  
  #-----------------------------------------------------------
  # score results
  if(length(true.idx) > 0) {
    
    true.len = length(true.idx)
    res$true.idx = true.idx
    
    ## vip
    tp = length(which(vip.imp.cols %in% true.idx))
    positive.len = length(vip.imp.cols)
    
    res$vip.precision = (tp * 1.0) / (positive.len * 1.0)
    res$vip.recall = (tp * 1.0) / (true.len * 1.0)
    res$vip.f1 = 2 * res$vip.precision * res$vip.recall / (res$vip.precision + res$vip.recall)
    
    ## mi
    tp = length(which(mi.imp.cols %in% true.idx))
    positive.len = length(mi.imp.cols)
    
    res$mi.precision = (tp * 1.0) / (positive.len * 1.0)
    res$mi.recall = (tp * 1.0) / (true.len * 1.0)
    res$mi.f1 = 2 * res$mi.precision * res$mi.recall / (res$mi.precision + res$mi.recall)
    
    if (length(categorical.idx) > 0) {
      ## within-type vip
      tp = length(which(within.type.vip.imp.cols %in% true.idx))
      positive.len = length(within.type.vip.imp.cols)
      
      res$wt.vip.precision = (tp * 1.0) / (positive.len * 1.0)
      res$wt.vip.recall = (tp * 1.0) / (true.len * 1.0)
      res$wt.vip.f1 = 2 * res$wt.vip.precision * res$wt.vip.recall / (res$wt.vip.precision + res$wt.vip.recall)
    }
  }
  
  #-----------------------------------------------------------
  if (plot) {
    
    if ((n.var.plot == Inf) | (n.var.plot > ncol(x.train))){
      n.var.plot = ncol(x.train)
    }
    
    ## vip
    non.zero.idx = which(avg.vip > 0)[1:min(n.var.plot, length(which(avg.vip > 0)))]
    plot.n = length(non.zero.idx)
    if(length(non.zero.idx) < length(avg.vip)) 
      warning(paste(length(which(avg.vip == 0)), "predictors with inclusion proportions of 0 omitted from plots."))
    max.cut = max(apply(permute.vips, 2, quantile, probs = 1-alpha, na.rm = TRUE))
    
    plot(1:plot.n, avg.vip[non.zero.idx], 
         type = "n", xlab = "Predictors", xaxt = "n", ylim = c(0, max(max(avg.vip), max.cut * 1.1)),
         main = "Permutation-Based Variable Selection", ylab = "BART VIP")
    axis(1, at = 1:plot.n, labels = names(avg.vip[non.zero.idx]), las = 2)
    for (j in non.zero.idx){
      points(j, avg.vip[j], 
             pch = ifelse(avg.vip[j] <= quantile(permute.vips[, j], 1 - alpha), 1, 16),
             col = ifelse(names(avg.vip[j]) %in% categorical.names, 'green', 'red'))
    }
    sapply(non.zero.idx, function(s){segments(s, 0, x1 = s, quantile(permute.vips[, s], 1 - alpha), col = "grey")})
    
    ## mi
    non.zero.idx = which(median.mi > 0)[1 : min(n.var.plot, length(which(median.mi > 0)))]
    plot.n = length(non.zero.idx)
    if(length(non.zero.idx) < length(median.mi)) 
      warning(paste(length(which(median.mi == 0)), 
                    "predictors with Metropolis importance of 0 omitted from plots."))
    max.cut = max(apply(permute.mis, 2, quantile, probs = 1 - alpha, na.rm = TRUE))
    
    plot(1:plot.n, median.mi[non.zero.idx], 
         type = "n", xlab = "Predictors", xaxt = "n", ylim = c(0, max(max(median.mi), max.cut * 1.1)),
         main = "Permutation-Based Variable Selection", ylab = "BART MI")
    axis(1, at = 1:plot.n, labels = names(median.mi[non.zero.idx]), las = 2)
    for (j in non.zero.idx){
      points(j, median.mi[j], 
             pch = ifelse(median.mi[j] <= quantile(permute.mis[, j], 1 - alpha), 1, 16),
             col = ifelse(names(median.mi[j]) %in% categorical.names, 'green', 'red'))
    }
    sapply(non.zero.idx, function(s){segments(s, 0, x1 = s, quantile(permute.mis[, s], 1 - alpha), col = "grey")})
    
    if (length(categorical.idx) > 0) {
      ## within-type vip
      non.zero.idx = which(avg.within.type.vip > 0)[1:min(n.var.plot, length(which(avg.within.type.vip > 0)))]
      plot.n = length(non.zero.idx)
      if(length(non.zero.idx) < length(avg.within.type.vip)) 
        warning(paste(length(which(avg.within.type.vip == 0)), 
                      "predictors with inclusion proportions (within-type) of 0 omitted from plots."))
      max.cut = max(apply(permute.within.type.vips, 2, quantile, probs = 1 - alpha, na.rm = TRUE))
      
      plot(1:plot.n, avg.within.type.vip[non.zero.idx], 
           type = "n", xlab = "Predictors", xaxt = "n", ylim = c(0, max(max(avg.within.type.vip), max.cut * 1.1)),
           main = "Permutation-Based Variable Selection", ylab = "BART Within-Type VIP")
      axis(1, at = 1:plot.n, labels = names(avg.within.type.vip[non.zero.idx]), las = 2)
      for (j in non.zero.idx){
        points(j, avg.within.type.vip[j], 
               pch = ifelse(avg.within.type.vip[j] <= quantile(permute.within.type.vips[, j], 1 - alpha), 1, 16),
               col = ifelse(names(avg.within.type.vip[j]) %in% categorical.names, 'green', 'red'))
      }
      sapply(non.zero.idx, function(s)
        {segments(s, 0, x1 = s, quantile(permute.within.type.vips[, s], 1 - alpha), col = "grey")})
    }
  }
  
  #------------------------------
  # timer ends
  end = Sys.time()
  if(verbose) cat("Elapsed", end-start, '\n')
  
  return(res)
}

