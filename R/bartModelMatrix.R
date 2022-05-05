## This file is modified from the source file of the function BART::bartModelMatrix().
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

#' Create a matrix out of a vector or data frame
#' 
#' The external BART functions (e.g. \code{wbart()}) operate on matrices in memory. Therefore, if the user submits a vector or 
#' data frame, then this function converts it to a matrix. Also, it determines the number of cut points necessary for each column
#' when asked to do so. This function is inherited from the CRAN package 'BART'.
#' 
#' @param X A vector or data frame where the matrix is created.
#' @param numcut The maximum number of cut points to consider. If \code{numcut=0}, then return a matrix; otherwise, return a list
#' containing a matrix \code{X}, a vector \code{numcut} and a list \code{xinfo}.
#' @param usequants A Boolean argument indicating the way to generate cut points. If \code{usequants=FALSE}, then the cut points 
#' in \code{xinfo} are generated uniformly; otherwise, the quantiles are used for the cut points.
#' @param type An integer between \eqn{1} and \eqn{9} determining which algorithm is employed in the function \code{quantile()}.
#' @param rm.const A Boolean argument indicating whether to remove constant variables.
#' @param cont A Boolean argument indicating whether to assume all variables are continuous.
#' @param xinfo A list (matrix) where the items (rows) are the predictors and the contents (columns) of the items are the cut points.
#' If \code{xinfo=NULL}, BART will choose \code{xinfo} for the user.
#' 
#' @return The function \code{bartModelMatrix()} returns a list with the following components.
#' \item{X}{A matrix with rows corresponding to observations and columns corresponding to predictors (after dummification).}
#' \item{numcut}{A vector of \code{ncol(X)} integers with each indicating the number of cut points for the corresponding predictor.}
#' \item{rm.const}{A vector of indicators for the predictors (after dummification) used in BART; when the indicator is negative, 
#' it refers to remove that predictor.}
#' \item{xinfo}{A list (matrix) where the items (rows) are the predictors and the contents (columns) of the items are the cut points.}
#' \item{grp}{A vector of group indices for predictors. For example, if \eqn{2} appears \eqn{3} times in \code{grp}, the second 
#' predictor of \code{X} is a categorical predictor with \eqn{3} levels.}
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
#'   
#' Sparapani, R., Spanbauer, C. and McCulloch, R. (2021).
#'   "Nonparametric machine learning and efficient computation with bayesian additive regression trees: the BART R package."
#'   \emph{J. Stat. Softw.} \strong{97} 1--66.
#' @seealso 
#' \code{\link{wbart}} and \code{\link{pbart}}.
#' @examples  
#' ## simulate data (Scenario C.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, FALSE)
#' ## test bartModelMatrix() function
#' res = bartModelMatrix(data$X, numcut=100, usequants=FALSE, cont=FALSE, rm.const=TRUE)
bartModelMatrix = function(X, 
                           numcut=0L, 
                           usequants=FALSE, 
                           type=7,
                           rm.const=FALSE, 
                           cont=FALSE, 
                           xinfo=NULL) {
  
  X.class = class(X)[1]
  
  if(X.class == 'factor') {
    X.class = 'data.frame'
    X = data.frame(X = X)
  }
  
  grp = NULL
  
  if(X.class == 'data.frame') {
    p = dim(X)[2]
    xnm = names(X)
    for(i in 1:p) {
      if(is.factor(X[[i]])) {
        Xtemp = class.ind(X[[i]])
        colnames(Xtemp) = paste(xnm[i], 1:ncol(Xtemp), sep = '')
        X[[i]] = Xtemp
        grp = c(grp, rep(i, ncol(Xtemp)))
      } else {
        X[[i]] = cbind(X[[i]])
        colnames(X[[i]]) = xnm[i]
        grp = c(grp, i)
      }
    }
    Xtemp = cbind(X[[1]])
    if(p > 1) for(i in 2:p) Xtemp = cbind(Xtemp, X[[i]])
    X = Xtemp
  }
  else if(X.class == 'numeric' | X.class == 'integer') {
    X = cbind(as.numeric(X))
    grp = 1
  }
  else if(X.class == 'NULL') return(X)
  else if(X.class != 'matrix')
    stop('Expecting either a factor, a vector, a matrix or a data.frame')
  
  N = nrow(X)
  p = ncol(X)
  
  xinfo. = matrix(nrow=p, ncol=numcut)
  nc = numcut
  rm.vars = c()
  
  if(N > 0 & p > 0 & (rm.const | numcut[1] > 0)) {
    for(j in 1:p) {
      X.class = class(X[1, j])[1]
      
      if(X.class == 'numeric' | X.class == 'integer') {
        xs = unique(sort(X[ , j]))
        k = length(xs)
        nc[j] = numcut
        
        if(k %in% 0:1) {
          rm.vars = c(rm.vars, -j)
          nc[j] = 1
          if(k == 0) xs = NA
        }
        else if(cont) 
          xs = seq(xs[1], xs[k], length.out = numcut + 2)[-c(1, numcut + 2)]
        else if(k < numcut) {
          xs = 0.5*(xs[1:(k-1)] + xs[2:k])
          nc[j] = k - 1
        }
        else if(usequants) {
          xs = quantile(X[ , j], type = type,
                        probs = (0:(numcut + 1)) / (numcut + 1))[-c(1, numcut + 2)]
          names(xs) = NULL
        }
        else xs = seq(xs[1], xs[k], length.out = numcut + 2)[-c(1, numcut + 2)]
      }
      else
        stop(paste0('Variables of type ', X.class, ' are not supported'))
      
      xinfo.[j, 1:nc[j] ] = xs
    }
  }
  
  X = data.matrix(X)
  
  if(length(xinfo) > 0) {
    if(is.list(xinfo)) for(j in 1:p) xinfo.[j, 1:length(xinfo[[j]])] = xinfo[[j]]
    else if(is.matrix(xinfo)) xinfo. = xinfo
    else stop('Only a list or a matrix can be provided for xinfo')
    
    for(j in 1:p) nc[j] = sum(!is.na(xinfo.[j, ]))
  }
  
  xinfo = xinfo.
  
  if(rm.const & length(rm.vars) > 0) {
    X = X[ , rm.vars]
    nc = nc[rm.vars]
    xinfo = xinfo[rm.vars, ]
  }
  else if(length(rm.vars) == 0) rm.vars = 1:p
  
  dimnames(xinfo) = list(dimnames(X)[[2]], NULL)
  
  if(numcut == 0) return(X)
  else return(list(X = X, numcut = as.integer(nc), rm.const = rm.vars,
                   xinfo = xinfo, grp = grp))
}