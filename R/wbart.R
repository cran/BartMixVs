## This file is modified from the source file of the function BART::wbart().
## See below for the copyright of the CRAN R package 'BART'.

## BART: Bayesian Additive Regression Trees
## Copyright (C) 2018 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

#' BART for continuous responses
#' 
#' BART is a Bayesian approach to nonparametric function estimation and inference using a sum of trees.\cr
#' For a continuous response \eqn{y} and a \eqn{p-}dimensional vector of predictors \eqn{x = (x_1, ..., x_p)'}, 
#' BART models \eqn{y} and \eqn{x} using \deqn{y = f(x) + \epsilon,}
#' where \eqn{f} is a sum of Bayesian regression trees function and \eqn{\epsilon ~ N(0, \sigma^2)}.\cr
#' The function \code{wbart()} is inherited from the CRAN R package 'BART' and two modifications are made 
#' for the splitting probability and variable importance (see Details).
#' 
#' This function is inherited from \code{BART::wbart()}.
#' While the original features of \code{BART::wbart()} are preserved, two modifications are made.\cr
#' The first modification is to provide two types of split probability for BART. One split probability is proposed in
#' Chipman et al. (2010) and defined as \deqn{p(d) = \gamma * (1+d)^{-\beta},} where \eqn{d} is the depth of the node,
#' \eqn{\gamma \in (0,1)} and \eqn{\beta \in (0,\infty)}. The other split probability is proposed by Rockova and Saha (2019) 
#' and defined as \deqn{p(d) = \gamma^d,} where \eqn{\gamma \in (1/n, 1/2)}. BART with the second split probability is proved
#' to achieve the optimal posterior contraction.\cr
#' The second modification is to provide five types of variable importance measures (\code{vip}, \code{within.type.vip},
#' \code{pvip}, \code{varprob.mean} and \code{mi}) in the return object, for the sake of the existence of mixed-type predictors.
#' 
#' @param x.train A matrix or a data frame of predictors values (for training) with each row corresponding to an 
#' observation and each column corresponding to a predictor. If a predictor is a factor with \eqn{q} levels in a data frame,
#' it is replaced with \eqn{q} dummy variables.
#' @param y.train A vector of continuous response values for training.
#' @param x.test A matrix or a data frame of predictors values for testing, which has the same structure as \code{x.train}.
#' @param sparse A Boolean argument indicating whether to replace the discrete uniform distribution for selecting a split 
#' variable with a categorical distribution whose event probabilities follow a Dirichlet distribution 
#' (see Linero (2018) for details).
#' @param theta Set \code{theta} parameter; zero means random.
#' @param omega Set \code{omega} parameter; zero means random.
#' @param a A sparse parameter of \eqn{Beta(a, b)} hyper-prior where \eqn{0.5<=a<=1}; a lower value induces more sparsity.
#' @param b A sparse parameter of \eqn{Beta(a, b)} hyper-prior; typically, \eqn{b=1}.
#' @param augment A Boolean argument indicating whether data augmentation is performed in the variable selection procedure 
#' of Linero (2018).
#' @param rho A sparse parameter; typically \eqn{\rho = p} where \eqn{p} is the number of predictors.
#' @param xinfo A matrix of cut-points with each row corresponding to a predictor and each column corresponding to a cut-point.
#' \code{xinfo=matrix(0.0,0,0)} indicates the cut-points are specified by BART.
#' @param usequants A Boolean argument indicating how the cut-points in \code{xinfo} are generated; 
#' If \code{usequants=TRUE}, uniform quantiles are used for the cut-points; Otherwise, the cut-points are generated uniformly.
#' @param numcut The number of possible cut-points; If a single number is given, this is used for all predictors; 
#' Otherwise a vector with length equal to \code{ncol(x.train)} is required, where the \eqn{i-}th element gives the number of 
#' cut-points for the \eqn{i-}th predictor in \code{x.train}. If \code{usequants=FALSE}, \code{numcut} equally spaced 
#' cut-points are used to cover the range of values in the corresponding column of \code{x.train}. 
#' If \code{usequants=TRUE}, then min(\code{numcut}, the number of unique values in the corresponding column of 
#' \code{x.train} - 1) cut-point values are used.
#' @param cont A Boolean argument indicating whether to assume all predictors are continuous.
#' @param rm.const A Boolean argument indicating whether to remove constant predictors.
#' @param grp A vector of group indices for predictors. For example, if \eqn{2} appears \eqn{3} times in \code{grp}, the second 
#' predictor of \code{x.train} is a categorical predictor with \eqn{3} levels. \code{grp} is required if \code{transposed=TRUE}.
#' @param xnames Column names of \code{x.train}. \code{xnames} is required if \code{transposed=TRUE}.
#' @param categorical.idx A vector of the column indices of categorical predictors in \code{x.train}. \code{categorical.idx} 
#' is required if \code{transposed=TRUE}.
#' @param power The power parameter of the polynomial splitting probability for the tree prior. Only used if 
#' \code{split.prob="polynomial"}.
#' @param base The base parameter of the polynomial splitting probability for the tree prior if \code{split.prob="polynomial"}; 
#' if \code{split.prob="exponential"}, the probability of splitting a node at depth \eqn{d} is \code{base}\eqn{^d}. 
#' @param split.prob A string indicating what kind of splitting probability is used for the tree prior. If 
#' \code{split.prob="polynomial"}, the splitting probability in Chipman et al. (2010) is used; 
#' If \code{split.prob="exponential"}, the splitting probability in Rockova and Saha (2019) is used.
#' @param k The number of prior standard deviations that \eqn{E(Y|x) = f(x)} is away from \eqn{+/-.5}. The response 
#' (\code{y.train}) is internally scaled to the range from \eqn{-.5} to \eqn{.5}. The bigger \code{k} is, the more conservative 
#' the fitting will be.
#' @param sigmaf The standard deviation of \code{f}.
#' @param sigest A rough estimate of the error standard deviation, the square of which follows an inverse chi-squared prior. 
#' If \code{sigest=NA}, the rough estimate will be the usual least square estimator; Otherwise, the supplied value will be used.
#' @param sigdf The degrees of freedom for the error variance prior.
#' @param sigquant The quantile of the error variance prior, where \code{sigest} is placed. The closer the quantile is to 1, 
#' the more aggressive the fit will be.
#' @param lambda The scale parameter of the error variance prior.
#' @param fmean BART operates on \code{y.train} centered by \code{fmean}.
#' @param w A vector of weights which multiply the standard deviation.
#' @param ntree The number of trees in the ensemble.
#' @param ndpost The number of posterior samples returned.
#' @param nskip The number of posterior samples burned in.
#' @param keepevery Every \code{keepevery} posterior sample is kept to be returned to the user.
#' @param nkeeptrain The number of posterior samples returned for the train data.
#' @param nkeeptest The number of posterior samples returned for the test data.
#' @param nkeeptestmean The number of posterior samples returned for the test mean.
#' @param nkeeptreedraws The number of posterior samples returned for the tree draws.
#' @param printevery As the MCMC runs, a message is printed every \code{printevery} iterations.
#' @param transposed A Boolean argument indicating whether the matrices \code{x.train} and \code{x.test} are transposed.
#' @param verbose A Boolean argument indicating whether any messages are printed out.
#' 
#' @return The function \code{wbart()} returns an object of type \code{wbart} which essentially is a list consisting of the 
#' following components.
#' \item{sigma}{A vector with \code{nskip+ndpost*keepevery} posterior samples of \eqn{\sigma}.}
#' \item{yhat.train.mean}{\code{colMeans(yhat.train)}.}
#' \item{yhat.train}{A matrix with \code{ndpost} rows and \code{nrow(x.train)} columns with each row corresponding to a draw 
#' \eqn{f*} from the posterior of \eqn{f} and each column corresponding to a training data point. The \eqn{(i,j)}-th element of
#' the matrix is \eqn{f*(x)} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th training data point. Burn-in posterior 
#' samples are dropped.}
#' \item{yhat.test.mean}{\code{colMeans(yhat.test)}.}
#' \item{yhat.test}{A matrix with \code{ndpost} rows and \code{nrow(x.test)} columns with each row corresponding to a draw 
#' \eqn{f*} from the posterior of \eqn{f} and each column corresponding to a test data point. The \eqn{(i,j)}-th element of
#' the matrix is \eqn{f*(x)} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th test data point. Burn-in posterior 
#' samples are dropped.}
#' \item{varcount}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the number of times that the \eqn{j}-
#' th predictor is used as a split variable in the \eqn{i}-th posterior sample.}
#' \item{varprob}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the split probability of the \eqn{j}-
#' th predictor in the \eqn{i}-th posterior sample. Only useful when DART is fit, i.e., \code{sparse=TRUE}.}
#' \item{treedraws}{A list containing the posterior samples of the ensembles (trees structures, split variables and split values);
#' Can be used for prediction.}
#' \item{proc.time}{The process time of running the function \code{wbart()}.}
#' \item{mu}{BART operates on \code{y.train} centered by \code{fmean}.}
#' \item{mr.vecs}{A list of \eqn{ncol(x.train)} sub-lists with each corresponding to a predictor; Each sub-list contains 
#' \code{ndpost} vectors with each vector containing the (birth) Metropolis ratios for splits using the predictor as the split 
#' variable in that posterior sample.}
#' \item{vip}{A vector of variable inclusion proportions (VIP) proposed in Chipman et al. (2010).}
#' \item{within.type.vip}{A vector of within-type VIPs proposed in Luo and Daniels (2021).}
#' \item{pvip}{A vector of marginal posterior variable inclusion probabilities (PVIP) proposed in Linero (2018); Only useful
#' when DART is fit, i.e., \code{sparse=TRUE}.}
#' \item{varprob.mean}{A vector of posterior split probabilities (PSP) proposed in Linero (2018); Only useful when DART is fit, 
#' i.e., \code{sparse=TRUE}.}
#' \item{mr.mean}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the average Metropolis acceptance ratio
#' per splitting rule using the \eqn{j}-th predictor in the \eqn{i}-th posterior sample.}
#' \item{mi}{A vector of Metropolis importance (MI) proposed in Luo and Daniels (2021).}
#' \item{rm.const}{A vector of indicators for the predictors (after dummification) used in BART; when the indicator is negative, 
#' it refers to remove that predictor.}
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
#'   
#' Sparapani, R., Spanbauer, C. and McCulloch, R. (2021).
#'   "Nonparametric machine learning and efficient computation with bayesian additive regression trees: the BART R package."
#'   \emph{J. Stat. Softw.} \strong{97} 1--66.
#' @seealso 
#' \code{\link{mc.wbart}}, \code{\link{pbart}} and \code{\link{pwbart}}.
#' @examples  
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## test wbart() function
#' res = wbart(data$X, data$Y, ntree=10, nskip=100, ndpost=100)
wbart = function(x.train, 
                 y.train, 
                 x.test=matrix(0.0,0,0),
                 sparse=FALSE, 
                 theta=0, 
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
                 grp=NULL, 
                 xnames=NULL, 
                 categorical.idx=NULL,
                 power=2.0, 
                 base=-1.0, 
                 split.prob="polynomial",
                 k=2.0, 
                 sigmaf=NA, 
                 sigest=NA, 
                 sigdf=3, 
                 sigquant=.90, 
                 lambda=NA,
                 fmean=mean(y.train), 
                 w=rep(1,length(y.train)),
                 ntree=200L, 
                 ndpost=1000L, 
                 nskip=1000L, 
                 keepevery=1L,
                 nkeeptrain=ndpost, 
                 nkeeptest=ndpost, 
                 nkeeptestmean=ndpost, 
                 nkeeptreedraws=ndpost,
                 printevery=100L, 
                 transposed=FALSE, 
                 verbose=FALSE
                 ) {
  #--------------------------------------------------
  #data
  n = length(y.train)
  
  if(!transposed) {
    if(is.null(dim(x.train))) {
      xnames = "X"
    } else {
      xnames = dimnames(x.train)[[2]]
    }
    
    temp = bartModelMatrix(x.train, numcut, usequants = usequants,
                           cont = cont, xinfo = xinfo, rm.const = rm.const)
    
    x.train = t(temp$X)
    numcut = temp$numcut
    xinfo = temp$xinfo
    if(length(x.test) > 0) {
      x.test = bartModelMatrix(x.test)
      x.test = t(x.test[ , temp$rm.const])
    }
    rm.const = temp$rm.const
    grp = temp$grp
    rm(temp)
    
    if(length(grp) == 0){
      p0 = nrow(x.train)
      grp = 1:p0
    } else {
      p0 = length(unique(grp))  # number of predictors before dummification
    }
    categorical.idx = unique(grp)[which(sapply(unique(grp), function(s) sum(s==grp)) > 1)]
    
  } else {
    if(any(length(rm.const) == 0, length(grp) == 0, length(xnames) == 0))
      stop('Did not provide rm.const, grp and xnames for x.train after transpose!')
    if(is.logical(rm.const))
      stop('Did not provide rm.const for x.train after transpose!')
    if((length(grp) > length(unique(grp))) & (length(categorical.idx) <= 0))
      stop('Did not provide categorical.idx for x.train that contains categorical predictors!')
    
    p0 = length(unique(grp))
  }
  
  if(n != ncol(x.train))
    stop('The length of y.train and the number of rows in x.train must be identical')
  
  p = nrow(x.train)
  np = ncol(x.test)
  if(length(rho) == 0) rho=p
  
  if(!(split.prob %in% c("polynomial", "exponential"))) {
    stop("split.prob is either polynomial or exponential.")
  } else {
    if(split.prob == "polynomial") {
      if(base < 0)
        base = 0.95
    }
    if(split.prob == "exponential") {
      power = -1.0
      if(base < 0)
        base = 0.5
    }
  }
  
  y.train = y.train - fmean
  
  #--------------------------------------------------
  #set nkeeps for thinning
  if((nkeeptrain != 0) & ((ndpost %% nkeeptrain) != 0)) {
    nkeeptrain = ndpost
    cat('*****nkeeptrain set to ndpost\n')
  }
  if((nkeeptest != 0) & ((ndpost %% nkeeptest) != 0)) {
    nkeeptest = ndpost
    cat('*****nkeeptest set to ndpost\n')
  }
  if((nkeeptestmean != 0) & ((ndpost %% nkeeptestmean) != 0)) {
    nkeeptestmean = ndpost
    cat('*****nkeeptestmean set to ndpost\n')
  }
  if((nkeeptreedraws != 0) & ((ndpost %% nkeeptreedraws) != 0)) {
    nkeeptreedraws = ndpost
    cat('*****nkeeptreedraws set to ndpost\n')
  }
  
  #--------------------------------------------------
  #prior
  nu = sigdf
  if(is.na(lambda)) {
    if(is.na(sigest)) {
      if(p < n) {
        df = data.frame(t(x.train), y.train)
        lmf = lm(y.train~., df)
        sigest = summary(lmf)$sigma
      } else {
        sigest = sd(y.train)
      }
    }
    qchi = qchisq(1.0 - sigquant, nu)
    lambda = (sigest * sigest * qchi) / nu #lambda parameter for sigma prior
  } else {
    sigest = sqrt(lambda)
  }
  
  if(is.na(sigmaf)) {
    tau = (max(y.train) - min(y.train)) / (2 * k * sqrt(ntree))
  } else {
    tau = sigmaf / sqrt(ntree)
  }
  
  #--------------------------------------------------
  #call c++ function
  ptm = proc.time()
  res = .Call("cwbart",
              n,  #number of observations in training data
              p,  #dimension of x
              np, #number of observations in test data
              x.train,   #pxn training data x
              y.train,   #pxn training data x
              x.test,   #p*np test data x
              ntree,
              numcut,
              ndpost*keepevery,
              nskip,
              power,
              base,
              tau,
              nu,
              lambda,
              sigest,
              w,
              sparse,
              theta,
              omega,
              grp,
              a,
              b,
              rho,
              augment,
              nkeeptrain,
              nkeeptest,
              nkeeptestmean,
              nkeeptreedraws,
              printevery,
              xinfo,
              verbose
  )
  
  res$proc.time = proc.time() - ptm
  
  #--------------------------------------------------
  #returns
  res$mu = fmean
  res$yhat.train.mean = res$yhat.train.mean + fmean
  res$yhat.train = res$yhat.train + fmean
  res$yhat.test.mean = res$yhat.test.mean + fmean
  res$yhat.test = res$yhat.test + fmean
  
  if(nkeeptreedraws > 0)
    names(res$treedraws$cutpoints) = xnames
  
  #--------------------------------------------------
  #importance
  if(length(grp) == length(unique(grp))) {
    ## no dummy variables
    dimnames(res$varcount)[[2]] = as.list(xnames)
    dimnames(res$varprob)[[2]] = as.list(xnames)
    
    ## vip: variable inclusion proportions
    res$vip = colMeans(t(apply(res$varcount, 1, function(s) s / sum(s))))
    
    ## (marginal) posterior variable inclusion probability
    res$pvip = colMeans(res$varcount > 0)
    
    ## posterior s_j's (only in DART)
    res$varprob.mean = colMeans(res$varprob)
    
    ## mi: Metropolis importance
    mr.vecs = lapply(res$mr_vecs, function(s) lapply(s, function(v) v[-1]))  # remove the meaningless first 0
    res$mr_vecs = NULL
    res$mr.vecs = mr.vecs
    mr.mean = matrix(unlist(lapply(mr.vecs, function(s) lapply(s, function(v) ifelse(length(v) > 0, mean(v), 0.0)))), 
                     ncol = p, byrow = TRUE)
    res$mr.mean = mr.mean
    res$mi = colMeans(t(apply(mr.mean, 1, function(s) s / sum(s))))
    names(res$mi) = as.list(xnames)
    dimnames(res$mr.mean)[[2]] = as.list(xnames)
  } else {
    ## merge importance scores for dummy variables
    varcount = matrix(NA, nrow = nkeeptreedraws, ncol = p0)
    varprob = matrix(NA, nrow = nkeeptreedraws, ncol = p0)
    
    mr.vecs = lapply(res$mr_vecs, function(s) list(s[[1]][-1]))
    #mr0_vecs = lapply(res$mr0_vecs, function(s) list(s[[1]][-1]))
    varcount[, 1] = res$varcount[, 1]
    varprob[, 1] = res$varprob[, 1]
    
    j = 1
    for (l in 2:p) {
      if (grp[l] == grp[l-1]) {
        varcount[, j] = varcount[, j] + res$varcount[, l]
        varprob[, j] = varprob[, j] + res$varprob[, l]
        for (i in 1:nkeeptreedraws) {
          mr.vecs[[i]][[j]] = c(mr.vecs[[i]][[j]], res$mr_vecs[[i]][[l]][-1])
          #mr0_vecs[[i]][[j]] = c(mr0_vecs[[i]][[j]], res$mr0_vecs[[i]][[l]][-1])
        }
      } else {
        j = j + 1
        varcount[, j] = res$varcount[, l]
        varprob[, j] = res$varprob[, l]
        for (i in 1:nkeeptreedraws) {
          mr.vecs[[i]][[j]] = res$mr_vecs[[i]][[l]][-1]
          #mr0_vecs[[i]][[j]] = res$mr0_vecs[[i]][[l]][-1]
        }
      }
    }
    
    dimnames(varcount)[[2]] = as.list(xnames)
    dimnames(varprob)[[2]] = as.list(xnames)
    
    res$varcount = varcount
    res$varprob = varprob
    res$mr.vecs = mr.vecs
    res$mr_vecs = NULL
    #res$mr0_vecs = mr0_vecs
    
    ## vip
    res$vip = colMeans(t(apply(varcount, 1, function(s) s / sum(s))))
    
    ## within-type vip
    within.type.vip = rep(0, p0)
    for (i in 1:nkeeptreedraws) {
      if (sum(varcount[i, categorical.idx]) != 0) {
        within.type.vip[categorical.idx] = within.type.vip[categorical.idx] + 
          varcount[i, categorical.idx] / sum(varcount[i, categorical.idx])
      }
      if (sum(varcount[i, -categorical.idx]) != 0) {
        within.type.vip[-categorical.idx] = within.type.vip[-categorical.idx] + 
          varcount[i, -categorical.idx] / sum(varcount[i, -categorical.idx])
      }
    }
    res$within.type.vip = within.type.vip / nkeeptreedraws
    names(res$within.type.vip) = xnames
    
    ## (marginal) posterior variable inclusion probability
    res$pvip = colMeans(varcount > 0)
    
    ## posterior s_j's (in DART)
    res$varprob.mean = colMeans(varprob)
    
    ## mi
    mr.mean = matrix(unlist(lapply(mr.vecs, function(s) lapply(s, function(v) ifelse(length(v) > 0, mean(v), 0.0)))), 
                     ncol = p0, byrow = TRUE)
    res$mr.mean = mr.mean
    res$mi = colMeans(t(apply(mr.mean, 1, function(s) s / sum(s))))
    dimnames(res$mr.mean)[[2]] = as.list(xnames)
    names(res$mi) = as.list(xnames)
  }
  
  res$rm.const = rm.const
  
  attr(res, 'class') = 'wbart'
  return(res)
}
