## This file is modified from the source file of the function BART::mc.pbart().
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

#' Probit BART for binary responses with parallel computation
#' 
#' BART is a Bayesian approach to nonparametric function estimation and inference using a sum of trees.\cr
#' For a binary response \eqn{y} and a \eqn{p-}dimensional vector of predictors \eqn{x = (x_1, ..., x_p)'}, 
#' probit BART models \eqn{y} and \eqn{x} using \deqn{P(Y=1|x) = \Phi[f(x)],}
#' where \eqn{\Phi} is the CDF of the standard normal distribution and \eqn{f} is a sum of Bayesian regression trees function.\cr
#' The function \code{mc.pbart()} is inherited from the CRAN R package 'BART' and is a variant of the function 
#' \code{pbart()} with parallel computation.
#' 
#' This function is inherited from \code{BART::mc.pbart()} and is a variant of the function \code{pbart()} with parallel computation.
#' While the original features of \code{BART::pbart()} are preserved, two modifications are made.\cr
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
#' @param binaryOffset The binary offset term in the probit BART model, i.e., \eqn{P(Y=1|x) = \Phi[f(x)+binaryOffset]}.
#' @param ntree The number of trees in the ensemble.
#' @param ndpost The number of posterior samples returned.
#' @param nskip The number of posterior samples burned in.
#' @param keepevery Every \code{keepevery} posterior sample is kept to be returned to the user.
#' @param printevery As the MCMC runs, a message is printed every \code{printevery} iterations.
#' @param keeptrainfits A Boolean argument indicating whether to keep \code{yhat.train} or not.
#' @param transposed A Boolean argument indicating whether the matrices \code{x.train} and \code{x.test} are transposed.
#' @param verbose A Boolean argument indicating whether any messages are printed out.
#' @param mc.cores The number of cores to employ in parallel.
#' @param nice Set the job niceness. The default niceness is \eqn{19} and niceness goes from \eqn{0} (highest) to \eqn{19} 
#' (lowest).
#' @param seed Seed required for reproducible MCMC.
#' 
#' @return The function \code{pbart()} returns an object of type \code{pbart} which essentially is a list consisting of the 
#' following components.
#' \item{yhat.train}{A matrix with \code{ndpost} rows and \code{nrow(x.train)} columns with each row corresponding to a draw 
#' \eqn{f*} from the posterior of \eqn{f} and each column corresponding to a training data point. The \eqn{(i,j)}-th element of
#' the matrix is \eqn{f*(x)+binaryOffset} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th training data point. 
#' Burn-in posterior samples are dropped.}
#' \item{prob.train}{A matrix with the same structure as \code{yhat.train} and the \eqn{(i,j)}-th element of \code{prob.train}
#' is \eqn{\Phi[f*(x)+binaryOffset]} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th training data point.}
#' \item{prob.train.mean}{A vector which is \code{colMeans(prob.train)}.}
#' \item{yhat.test}{A matrix with \code{ndpost} rows and \code{nrow(x.test)} columns with each row corresponding to a draw 
#' \eqn{f*} from the posterior of \eqn{f} and each column corresponding to a test data point. The \eqn{(i,j)}-th element of
#' the matrix is \eqn{f*(x)+binaryOffset} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th test data point. 
#' Burn-in posterior samples are dropped.}
#' \item{prob.test}{A matrix with the same structure as \code{yhat.test} and the \eqn{(i,j)}-th element of \code{prob.test}
#' is \eqn{\Phi[f*(x)+binaryOffset]} for the \eqn{i}-th kept draw of \eqn{f} and the \eqn{j}-th test data point.}
#' \item{prob.test.mean}{A vector which is \code{colMeans(prob.test)}.}
#' \item{varcount}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the number of times that the \eqn{j}-
#' th predictor is used as a split variable in the \eqn{i}-th posterior sample.}
#' \item{varprob}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the split probability of the \eqn{j}-
#' th predictor in the \eqn{i}-th posterior sample. Only useful when DART is fit, i.e., \code{sparse=TRUE}.}
#' \item{treedraws}{A list containing the posterior samples of the ensembles (trees structures, split variables and split values);
#' Can be used for prediction.}
#' \item{proc.time}{The process time of running the function \code{wbart()}.}
#' \item{vip}{A vector of variable inclusion proportions (VIP) proposed in Chipman et al. (2010).}
#' \item{within.type.vip}{A vector of within-type VIPs proposed in Luo and Daniels (2021).}
#' \item{pvip}{A vector of marginal posterior variable inclusion probabilities (PVIP) proposed in Linero (2018); Only useful
#' when DART is fit, i.e., \code{sparse=TRUE}.}
#' \item{varprob.mean}{A vector of posterior split probabilities (PSP) proposed in Linero (2018); Only useful when DART is fit, 
#' i.e., \code{sparse=TRUE}.}
#' \item{mr.vecs}{A list of \eqn{ncol(x.train)} sub-lists with each corresponding to a predictor; Each sub-list contains 
#' \code{ndpost} vectors with each vector containing the (birth) Metropolis ratios for splits using the predictor as the split 
#' variable in that posterior sample.}
#' \item{mr.mean}{A matrix with \code{ndpost} rows and \code{ncol(x.train)} columns with each row corresponding to a draw of 
#' the ensemble and each column corresponding to a predictor. The \eqn{(i,j)}-th element is the average Metropolis acceptance ratio
#' per splitting rule using the \eqn{j}-th predictor in the \eqn{i}-th posterior sample.}
#' \item{mi}{A vector of Metropolis importance (MI) proposed in Luo and Daniels (2021).}
#' \item{rm.const}{A vector of indicators for the predictors (after dummification) used in BART; when the indicator is negative, 
#' it refers to remove that predictor.}
#' \item{binaryOffset}{The binary offset term used in BART.}
#' \item{ndpost}{The number of posterior samples returned.}
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
#' \code{\link{pbart}}.
#' @examples  
#' ## simulate data (Scenario B.M.1. in Luo and Daniels (2021))
#' set.seed(123)
#' data = mixone(100, 10, 1, TRUE)
#' ## parallel::mcparallel/mccollect do not exist on windows
#' if(.Platform$OS.type=='unix') {
#' ## test mc.pbart() function
#'   res = mc.pbart(data$X, data$Y, ntree=10, nskip=100, ndpost=100, mc.cores=2)
#' }

mc.pbart = function(x.train, 
                    y.train, 
                    x.test=matrix(0.0, 0L, 0L),
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
                    k=2.0, ## BEWARE: do NOT use k for other purposes below
                    power=2.0, 
                    base=0.95,
                    split.prob="polynomial",
                    binaryOffset=NULL,
                    ntree=50L, 
                    ndpost=1000L, 
                    nskip=100L,
                    keepevery=1L, 
                    printevery=100L,
                    keeptrainfits=TRUE, 
                    transposed=FALSE,
                    verbose=FALSE,
                    mc.cores=2L, 
                    nice=19L,
                    seed=99L) {
    
    # data
    if(is.null(dim(x.train))) {
        xnames = "X"
    } else {
        xnames = dimnames(x.train)[[2]]  # predictor names before dummification
    }
    
    if(.Platform$OS.type!='unix')
        stop('parallel::mcparallel/mccollect do not exist on windows')
    
    RNGkind("L'Ecuyer-CMRG")
    set.seed(seed)
    parallel::mc.reset.stream()
    
    if(!transposed) {
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
        
        p0 = length(unique(grp))  # number of predictors before dummification
        categorical.idx = unique(grp)[which(sapply(unique(grp), 
                                                   function(s) sum(s == grp)) > 1)]
        
    }
    
    mc.cores.detected = detectCores()
    
    if(mc.cores > mc.cores.detected) mc.cores = mc.cores.detected
    
    mc.ndpost = ceiling(ndpost/mc.cores)
    
    for(i in 1:mc.cores) {
        parallel::mcparallel({psnice(value = nice);
            pbart(x.train = x.train, y.train = y.train, x.test = x.test,
                  sparse = sparse, theta = theta, omega = omega, a = a, b = b, augment = augment, rho = rho,
                  xinfo = xinfo, numcut = numcut, rm.const = rm.const, 
                  grp = grp, xnames = xnames, categorical.idx = categorical.idx,
                  k = k, power = power, base = base, split.prob = split.prob, binaryOffset = binaryOffset,
                  ntree = ntree, ndpost = mc.ndpost, nskip = nskip, keepevery = keepevery,
                  printevery = printevery, transposed = TRUE, verbose = verbose)},
            silent = (i != 1))
        ## to avoid duplication of output
        ## capture stdout from first posterior only
    }
    
    post.list = parallel::mccollect()
    
    post = post.list[[1]]
    
    if(mc.cores == 1 | attr(post, 'class') != 'pbart') return(post)
    else {
        if(class(rm.const)[1]!='logical') post$rm.const = rm.const
        
        post$ndpost = mc.cores * mc.ndpost
        
        p = nrow(x.train[post$rm.const, ])
        
        old.text = paste0(as.character(mc.ndpost), ' ', as.character(ntree), ' ', as.character(p))
        old.stop = nchar(old.text)
        
        post$treedraws$trees = sub(old.text,
                                   paste0(as.character(post$ndpost), ' ',
                                          as.character(ntree), ' ',
                                          as.character(p)),
                                   post$treedraws$trees)
        
        post$vip = (post$vip) * mc.ndpost
        post$pvip = (post$pvip) * mc.ndpost
        post$varprob.mean = (post$varprob.mean) * mc.ndpost
        post$mi = (post$mi) * mc.ndpost
        if (length(grp) > length(unique(grp)))
            post$within.type.vip = (post$within.type.vip) * mc.ndpost
        
        keeptestfits = length(x.test)>0
        
        for(i in 2:mc.cores) {
            if(keeptrainfits) {
                post$yhat.train = rbind(post$yhat.train, post.list[[i]]$yhat.train)
                post$prob.train = rbind(post$prob.train, post.list[[i]]$prob.train)
            }
            
            if(keeptestfits) {
                post$yhat.test = rbind(post$yhat.test, post.list[[i]]$yhat.test)
                post$prob.test = rbind(post$prob.test, post.list[[i]]$prob.test)
            }
            
            post$varcount = rbind(post$varcount, post.list[[i]]$varcount)
            post$varprob = rbind(post$varprob, post.list[[i]]$varprob)
            
            post$mr.vecs = c(post$mr.vecs, post.list[[i]]$mr.vecs)
            post$mr.mean = rbind(post$mr.mean, post.list[[i]]$mr.mean)
            
            post$vip = post$vip + (post.list[[i]]$vip) * mc.ndpost
            post$pvip = post$pvip + (post.list[[i]]$pvip) * mc.ndpost
            post$varprob.mean = post$varprob.mean + (post.list[[i]]$varprob.mean) * mc.ndpost
            post$mi = post$mi + (post.list[[i]]$mi) * mc.ndpost
            if (length(grp) > length(unique(grp)))
                post$within.type.vip = post$within.type.vip + (post.list[[i]]$within.type.vip) * mc.ndpost
            
            post$treedraws$trees = paste0(post$treedraws$trees,
                                          substr(post.list[[i]]$treedraws$trees, old.stop + 2,
                                                 nchar(post.list[[i]]$treedraws$trees)))
        }
        
        if(length(post$prob.train.mean) > 0)
            post$prob.train.mean = apply(post$prob.train, 2, mean)
        
        if(length(post$prob.test.mean)>0)
            post$prob.test.mean = apply(post$prob.test, 2, mean)
        
        # process importance
        dimnames(post$varcount)[[2]] = as.list(xnames)
        dimnames(post$varprob)[[2]] = as.list(xnames)
        dimnames(post$mr.mean)[[2]] = as.list(xnames)
        
        post$vip = post$vip / (mc.ndpost * mc.cores)
        post$pvip = post$pvip / (mc.ndpost * mc.cores)
        post$varprob.mean = post$varprob.mean / (mc.ndpost * mc.cores)
        post$mi = post$mi / (mc.ndpost * mc.cores)
        if (length(grp) > length(unique(grp)))
            post$within.type.vip = post$within.type.vip / (mc.ndpost * mc.cores)
        
        names(post$vip) = as.list(xnames)
        names(post$pvip) = as.list(xnames)
        names(post$varprob.mean) = as.list(xnames)
        names(post$mi) = as.list(xnames)
        if (length(grp) > length(unique(grp)))
            names(post$within.type.vip) = as.list(xnames)
        
        attr(post, 'class') = 'pbart'
        
        return(post)
    }
}
