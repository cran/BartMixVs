#' Variable selection with DART
#' 
#' This function implements the variable selection approach proposed in Linero (2018). Linero (2018) proposes DART, a variant of BART,
#' which replaces the discrete uniform distribution for selecting a split variable with a categorical distribution of which the event
#' probabilities follow a Dirichlet distribution. DART estimates the marginal posterior variable inclusion probability (MPVIP) for 
#' a predictor by the proportion of the posterior samples of the trees structures where the predictor is used as a split variable
#' at least once, and selects predictors with MPVIP at least \eqn{0.5}, yielding a median probability model.
#' 
#' See Linero (2018) or Section 2.2.3 in Luo and Daniels (2021) for details.\cr
#' If \code{vip.selection=TRUE}, this function also does variable selection by selecting variables whose BART VIP exceeds
#' \code{1/ncol{x.train}}.\cr
#' If \code{true.idx} is provided, the precision, recall and F1 scores are returned.\cr
#' If \code{plot=TRUE}, plots showing which predictors are selected are generated.
#' 
#' @param x.train A matrix or a data frame of predictors values with each row corresponding to an observation and each column 
#' corresponding to a predictor. If a predictor is a factor with \eqn{q} levels in a data frame, it is replaced with \eqn{q} dummy 
#' variables.
#' @param y.train A vector of response (continuous or binary) values.
#' @param probit A Boolean argument indicating whether the response variable is binary or continuous; \code{probit=FALSE} (by default)
#' means that the response variable is continuous. 
#' @param vip.selection A Boolean argument indicating whether to select predictors using BART VIPs.
#' @param true.idx (Optional) A vector of indices of the true relevant predictors; if provided, metrics including precision, recall
#' and F1 score are returned.
#' @param plot (Optional) A Boolean argument indicating whether plots are returned or not.
#' @param num.var.plot The number of variables to be plotted.
#' @param theta Set \code{theta} parameter; zero means random.
#' @param omega Set \code{omega} parameter; zero means random.
#' @param a A sparse parameter of \eqn{Beta(a, b)} hyper-prior where \eqn{0.5<=a<=1}; a lower value induces more sparsity.
#' @param b A sparse parameter of \eqn{Beta(a, b)} hyper-prior; typically, \eqn{b=1}.
#' @param augment A Boolean argument indicating whether data augmentation is performed in the variable selection procedure 
#' of Linero (2018).
#' @param rho A sparse parameter; typically \eqn{\rho = p} where \eqn{p} is the number of predictors.
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
#' @param printevery As the MCMC runs, a message is printed every \code{printevery} iterations.
#' @param verbose A Boolean argument indicating whether any messages are printed out.
#' 
#' @return The function \code{medianInclusion.vs()} returns two (or one if \code{vip.selection=FALSE}) plots if \code{plot=TRUE}
#' and a list with the following components.
#' \item{dart.pvip}{The vector of DART MPVIPs.}
#' \item{dart.pvip.imp.names}{The vector of column names of the predictors with DART MPVIP at least \eqn{0.5}.}
#' \item{dart.pvip.imp.cols}{The vector of column indices of the predictors with DART MPVIP at least \eqn{0.5}.}
#' \item{dart.precision}{The precision score for the DART approach; only returned if \code{true.idx} is provided.}
#' \item{dart.recall}{The recall score for the DART approach; only returned if \code{true.idx} is provided.}
#' \item{dart.f1}{The F1 score for the DART approach; only returned if \code{true.idx} is provided.}
#' \item{bart.vip}{The vector of BART VIPs; only returned if \code{vip.selection=TRUE}.}
#' \item{bart.vip.imp.names}{The vector of column names of the predictors with BART VIP exceeding \code{1/ncol{x.train}}; only 
#' returned if \code{vip.selection=TRUE}.}
#' \item{bart.vip.imp.cols}{The vector of column indicies of the predictors with BART VIP exceeding \code{1/ncol{x.train}}; only 
#' returned if \code{vip.selection=TRUE}.}
#' \item{bart.precision}{The precision score for the BART approach; only returned if \code{vip.selection=TRUE} and \code{true.idx} 
#' is provided.}
#' \item{bart.recall}{The recall score for the BART approach; only returned if \code{vip.selection=TRUE} and \code{true.idx} 
#' is provided.}
#' \item{bart.f1}{The F1 score for the BART approach; only returned if \code{vip.selection=TRUE} and \code{true.idx} 
#' is provided.}
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
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#'   
#' Rockova V, Saha E (2019). 
#'   “On theory for BART.” 
#'   \emph{In The 22nd International Conference on Artificial Intelligence and Statistics} (pp. 2839–2848). PMLR.
#' @seealso 
#' \code{\link{permute.vs}}, \code{\link{mc.backward.vs}} and \code{\link{abc.vs}}.
#' @examples 
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## test medianInclusion.vs() function
#' res = medianInclusion.vs(data$X, data$Y, probit=FALSE, vip.selection=TRUE,  
#' true.idx=c(1, 2, 6:8), plot=FALSE, ntree=10, ndpost=100, nskip=100, verbose=FALSE)
medianInclusion.vs = function(x.train, 
                               y.train, 
                               probit=FALSE,             ## pbart or wbart    
                               vip.selection=TRUE,       ## if do VS using BART VIP with threshold = 1/p
                               true.idx=NULL, 
                               plot=FALSE, 
                               num.var.plot=Inf,
                               theta=0,                  ## DART parameters
                               omega=1,                      
                               a=0.5, 
                               b=1, 
                               augment=FALSE, 
                               rho=NULL,
                               xinfo=matrix(0.0,0,0), 
                               numcut=100L,
                               usequants=FALSE, 
                               cont=FALSE, 
                               rm.const=TRUE, 
                               power=2.0, 
                               base=.95,                   ## depth parameters
                               split.prob="polynomial",
                               k=2.0,                      ## leaf parameters
                               ntree=20L, 
                               ndpost=1000L, 
                               nskip=1000L, 
                               keepevery=1L, 
                               printevery=100L,
                               verbose=FALSE) {
  
  res = list()
  
  #------------------------------
  # timer starts
  start = Sys.time()
  
  #------------------------------------------
  # fit bart and/or dart
  if (probit) {
    dart = pbart(x.train = x.train, y.train = y.train,
                 sparse = TRUE, theta = theta, omega = omega, a = a, b = b, augment = augment, rho = rho,
                 xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                 power = power, base = base, split.prob = split.prob, k = k, 
                 ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                 printevery = printevery, verbose = verbose)
    
    if (vip.selection) {
      bart = pbart(x.train = x.train, y.train = y.train, sparse = FALSE, 
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                   power = power, base = base, split.prob = split.prob, k = k, 
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                   printevery = printevery, verbose = verbose)
    }
  } else {
    dart = wbart(x.train = x.train, y.train = y.train,
                 sparse = TRUE, theta = theta, omega = omega, a = a, b = b, augment = augment, rho = rho,
                 xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                 power = power, base = base, split.prob = split.prob, k = k, 
                 ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                 printevery = printevery, verbose = verbose)
    
    if (vip.selection) {
      bart = wbart(x.train = x.train, y.train = y.train, sparse = FALSE, 
                   xinfo = xinfo, numcut = numcut, usequants = usequants, cont = cont, rm.const = rm.const, 
                   power = power, base = base, split.prob = split.prob, k = k, 
                   ntree = ntree, ndpost = ndpost, nskip = nskip, keepevery = keepevery, 
                   printevery = printevery, verbose = verbose)
    }
  }
  
  #------------------------------------------
  # get importance scores
  
  ## DART marginal posterior inclusion probability, threshold = 0.5
  dart.pvip = dart$pvip
  dart.pvip = sort(dart.pvip, decreasing = T)
  dart.pvip.imp.names = names(dart.pvip[dart.pvip > 0.5])
  dart.pvip.imp.cols = sapply(1:length(dart.pvip.imp.names),
                              function(x) {which(dart.pvip.imp.names[x] == colnames(x.train))})
  
  res$dart.pvip = dart.pvip
  res$dart.pvip.imp.names = dart.pvip.imp.names
  res$dart.pvip.imp.cols = dart.pvip.imp.cols
  
  rm(dart)
  
  if (vip.selection) {
    ## BART variable inclusion proportions, threshold = 1/p
    bart.vip = bart$vip
    bart.vip = sort(bart.vip, decreasing = T)
    bart.vip.imp.names = names(bart.vip[bart.vip > (1/dim(x.train)[2])])
    bart.vip.imp.cols = sapply(1:length(bart.vip.imp.names),
                               function(x) {which(bart.vip.imp.names[x] == colnames(x.train))})
    
    res$bart.vip = bart.vip
    res$bart.vip.imp.names = bart.vip.imp.names
    res$bart.vip.imp.cols = bart.vip.imp.cols
    
    rm(bart)
  }
  
  
  #------------------------------------------
  # score the results
  if (length(true.idx) > 0) {
    true.len = length(true.idx)
    
    dart.tp = length(which(dart.pvip.imp.cols %in% true.idx))
    dart.positive.len = length(dart.pvip.imp.cols)
    
    dart.prec = (dart.tp * 1.0) / (dart.positive.len * 1.0)
    dart.rec = (dart.tp * 1.0) / (true.len * 1.0)
    dart.f1 = 2 * dart.prec * dart.rec / (dart.prec + dart.rec)
    
    res$dart.precision = dart.prec
    res$dart.recall = dart.rec
    res$dart.f1 = dart.f1
    
    if(vip.selection) {
      bart.tp = length(which(bart.vip.imp.cols %in% true.idx))
      bart.positive.len = length(bart.vip.imp.cols)
      
      bart.prec = (bart.tp * 1.0) / (bart.positive.len * 1.0)
      bart.rec = (bart.tp * 1.0) / (true.len * 1.0)
      bart.f1 = 2 * bart.prec * bart.rec / (bart.prec + bart.rec)
      
      res$bart.precision = bart.prec
      res$bart.recall = bart.rec
      res$bart.f1 = bart.f1
    }
  }
  
  #------------------------------------------
  # plot
  if (plot) {
    categorical.idx = which(sapply(x.train, function(s) {is.factor(s)}))
    categorical.names = names(categorical.idx)
    
    if (num.var.plot == Inf | num.var.plot > ncol(x.train)){
      num.var.plot = ncol(x.train)
    }
    
    ## dart
    non.zero.idx = which(dart.pvip > 0)[1:min(num.var.plot, length(which(dart.pvip > 0)))]
    plot.n = length(non.zero.idx)
    if(length(non.zero.idx) < length(dart.pvip)) 
      warning(paste(length(which(dart.pvip == 0)), 
                    "predictors with marginal posterior variable inclusion probability of 0 omitted from plots."))
    
    plot(1:plot.n, dart.pvip[non.zero.idx], 
         type = "n", xlab = "Predictors", xaxt = "n", ylim = c(0, max(dart.pvip) * 1.1),
         ylab = "DART Posterior Variable Inclusion Probability")
    axis(1, at = 1:plot.n, labels = names(dart.pvip[non.zero.idx]), las = 2)
    for (j in non.zero.idx){
      points(j, dart.pvip[j],
             pch = ifelse(dart.pvip[j] <= 0.5, 1, 16),
             col = ifelse(names(dart.pvip[j]) %in% categorical.names, 'green', 'red'))
    }
    abline(h = 0.5, col = "grey", lty = 2)
    
    if (vip.selection) {
      ## bart
      non.zero.idx = which(bart.vip > 0)[1:min(num.var.plot, length(which(bart.vip > 0)))]
      plot.n = length(non.zero.idx)
      if(length(non.zero.idx) < length(bart.vip)) 
        warning(paste(length(which(bart.vip == 0)), 
                      "predictors with variable inclusion proprotion of 0 omitted from plots."))
      
      plot(1:plot.n, bart.vip[non.zero.idx], 
           type = "n", xlab = "Predictors", xaxt = "n", ylim = c(0, max(bart.vip) * 1.1),
           ylab = "BART VIP")
      axis(1, at = 1:plot.n, labels = names(bart.vip[non.zero.idx]), las = 2)
      for (j in non.zero.idx){
        points(j, bart.vip[j],
               pch = ifelse(bart.vip[j] <= (1/dim(x.train)[2]), 1, 16),
               col = ifelse(names(bart.vip[j]) %in% categorical.names, 'green', 'red'))
      }
      abline(h = 1 / dim(x.train)[2], col = "grey", lty = 2)
    }
  }
  
  #------------------------------
  # timer ends
  end = Sys.time()
  if(verbose) cat("Elapsed", end-start, '\n')
  
  return(res)
}