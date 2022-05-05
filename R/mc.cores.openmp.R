## This file is copied from the source file of the function BART::mc.cores.openmp().
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

#' Detecting OpenMP
#' 
#' This function is inherited from the CRAN R package 'BART' and was designed for OpenMP. For example, the 
#' \code{pwbart} function can use OpenMP or the 'parallel' R package for multi-threading. On UNIX/Unix-like systems, 
#' OpenMP, if available, is discovered at install time. However, we know of no GPL licensed code available to detect 
#' OpenMP on Windows (for Artistic licensed OpenMP detection code on Windows, see the Bioconductor R package 'rGADEM'). 
#' To determine whether OpenMP is available at run time, we provide the function documented here.
#' 
#' @return This function returns \eqn{0} when OpenMP is not available; otherwise, an integer greater than \eqn{0} is 
#' returned when OpenMP is available (\eqn{1} is returned unless you are running in a multi-threaded process)
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
#' \code{\link{pwbart}}.
#' @examples  
#' mc.cores.openmp()
mc.cores.openmp=function() .Call("mc_cores_openmp")