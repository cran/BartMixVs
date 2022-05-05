## This file is modified from the source file of the function BART::predict.pbart().
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

#' Predict new observations with a fitted BART model
#' 
#' BART is a Bayesian approach to nonparametric function estimation and inference using a sum of trees.\cr
#' For a binary response \eqn{y}, probit BART models \eqn{y} and \eqn{x} using \deqn{P(Y=1|x)=\Phi[f(x)],}
#' where \eqn{\Phi} is the CDF of the standard normal distribution and \eqn{f} is a sum of Bayesian regression 
#' trees function.\cr
#' This function uses S3 method for the class \code{pbart} and is inherited from the CRAN R package 'BART'.
#' 
#' @param object An object of class \code{pbart}, returned from the function \code{pbart()}.
#' @param newdata A matrix of predictors with rows corresponding to new observations.
#' @param mc.cores The number of threads to utilize.
#' @param openmp A Boolean argument dictating whether OpenMP is utilized for parallel processing. This depends on
#' whether OpenMP is available on your system which, by default, is verified with the function \code{mc.cores.openmp()}.
#' @param ... Other arguments passed on to the function \code{pwbart()}.
#' 
#' @return Returns a matrix of prediction for \code{newdata}, whose rows correspond to draws and columns correspond to 
#' observations.
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
#' \code{\link{pwbart}} and \code{\link{pbart}}.
#' @examples  
#' ## simulate data (Scenario B.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, TRUE)
#' ## run pbart() function
#' res = pbart(data$X, data$Y, ntree=10, nskip=100, ndpost=100)
#' ## test predict.pbart() function
#' newdata = mixone(5, 10, 1, TRUE)$X
#' pred = predict(res, newdata)
predict.pbart <- function(object, newdata, mc.cores=1, openmp=(mc.cores.openmp()>0), ...) {

    # p <- length(object$treedraws$cutpoints)
    # 
    # if(p!=ncol(newdata))
    #     stop(paste0('The number of columns in newdata must be equal to ', p))

    if(.Platform$OS.type == "unix") mc.cores.detected = detectCores()
    else mc.cores.detected = NA

    if(!is.na(mc.cores.detected) && mc.cores>mc.cores.detected) mc.cores = mc.cores.detected

    if(.Platform$OS.type != "unix" || openmp || mc.cores==1) call = pwbart
    else call = mc.pwbart

    if(length(object$binaryOffset)==0) object$binaryOffset=object$offset

    pred = list(yhat.test=call(newdata, object$treedraws, object$rm.const, mc.cores=mc.cores,
                               mu=object$binaryOffset, ...))

    pred$prob.test = pnorm(pred$yhat.test)
    pred$prob.test.mean = apply(pred$prob.test, 2, mean)
    pred$binaryOffset = object$binaryOffset
    attr(pred, 'class') = 'pbart'

    return(pred)
}

