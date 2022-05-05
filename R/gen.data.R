#' Generate data for an example of Friedman (1991)
#' 
#' Generate data including responses and predictors values according to an example of Friedman, J. H. (1991). "Multivariate adaptive 
#' regression splines." \emph{Ann. Statist.} \strong{19} 1--141.
#' 
#' Sample the predictors \eqn{x_1, ..., x_p} from Uniform(0, 1) independently.
#' If \code{binary = FALSE}, sample the continuous response \eqn{y} from Normal(\eqn{f0(x), \sigma^2}), where 
#' \deqn{f0(x) = 10sin(\pi x_1*x_2) + 20(x_3-0.5)^2 + 10x_4 + 5x_5.}
#' If \code{binary = TRUE}, sample the binary response \eqn{y} from Bernoulli(\eqn{\Phi(f0(x))}) where \eqn{f0} is defined above and 
#' \eqn{\Phi} is the cumulative density function of the standard normal distribution.
#' 
#' @param n The number of observations.
#' @param p The number of predictors.
#' @param sigma The error variance.
#' @param binary A boolean argument: \code{binary = TRUE} indicates that binary responses are generated and \code{binary = FALSE} 
#' indicates that continuous responses are generated.
#' @return Return a list with the following components.
#' \item{X}{An n by p data frame representing predictors values, with each row corresponding an observation.}
#' \item{Y}{A vector of length n representing response values.}
#' \item{f0}{A vector of length n representing the values of \eqn{f0(x)}.}
#' \item{sigma}{The error variance which is only returned when \code{binary = FALSE}.}
#' \item{prob}{A vector of length n representing the values of \eqn{\Phi(f0(x))}, which is only returned when \code{binary = TRUE}.}
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Friedman, J. H. (1991). 
#'   "Multivariate adaptive regression splines." 
#'   \emph{Ann. Statist.} \strong{19} 1--141.
#'   
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#' @examples
#' data = friedman(100, 10, 1, FALSE)
friedman = function(n, p, sigma, binary) {
  
  X = matrix(runif(n * p), nrow = n, ncol = p)
  f0 = 10 * sin(pi * X[, 1] * X[,2]) + 20 * (X[,3] - 0.5)^2 + 10 * X[,4] + 5 * X[,5]
  X = as.data.frame(X)
  dimnames(X)[[2]] = sapply(1 : p, function(s) paste("X", s, sep = ""))
  
  if (binary) {
    f0 = scale(f0, center = T, scale = F)
    prob = pnorm(f0, mean = 0, sd = 1)
    Y = c()
    for (i in 1 : n) {
      Y[i] = rbinom(1, 1, prob = prob[i])
    }
    return(list(X = X, Y = Y, f0 = f0, prob = prob))
    
  } else {
    Y = rnorm(n, mean = f0, sd = sigma)
    return(list(X = X, Y = Y, f0 = f0, sigma = sigma))
  }
  
}


#' Generate data for an example of Zhu, Zeng and Kosorok (2015)
#' 
#' Generate data including responses and predictors values according to an example of Zhu, R., Zeng, D. and Kosorok, M. R. (2015). 
#' "Reinforcement learning trees." \emph{J. Amer. Statist. Assoc.} \strong{110} 1770--1784. 
#' 
#' Sample the predictors \eqn{x_1, ..., x_p} from Normal(\eqn{0, \Sigma}) with \eqn{\Sigma_{jk} = 0.3^{|j-k|}}, \eqn{j,k = 1, ..., p}.
#' If \code{binary = FALSE}, sample the continuous response \eqn{y} from Normal(\eqn{f0(x), \sigma^2}), where 
#' \deqn{f0(x) = 2x_1*x_4 + 2x_7*x_{10}.}
#' If \code{binary = TRUE}, sample the binary response \eqn{y} from Bernoulli(\eqn{\Phi(f0(x))}) where \eqn{f0} is defined above and
#'  \eqn{\Phi} is the cumulative density function of the standard normal distribution.
#' 
#' @param n The number of observations.
#' @param p The number of predictors.
#' @param sigma The error variance.
#' @param binary A boolean argument: \code{binary = TRUE} indicates that binary responses are generated and \code{binary = FALSE} 
#' indicates that continuous responses are generated.
#' @return Return a list with the following components.
#' \item{X}{An n by p data frame representing predictors values, with each row corresponding an observation.}
#' \item{Y}{A vector of length n representing response values.}
#' \item{f0}{A vector of length n representing the values of \eqn{f0(x)}.}
#' \item{sigma}{The error variance which is only returned when \code{binary = FALSE}.}
#' \item{prob}{A vector of length n representing the values of \eqn{\Phi(f0(x))}, which is only returned when \code{binary = TRUE}.}
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#'   
#' Zhu, R., Zeng, D. and Kosorok, M. R. (2015). 
#'   "Reinforcement learning trees." 
#'   \emph{J. Amer. Statist. Assoc.} \strong{110} 1770--1784.
#' @examples
#' data = checkerboard(100, 10, 1, FALSE)
checkerboard = function(n, p, sigma, binary) {
  
  cov_mtx = matrix(NA, nrow = p, ncol = p)
  for (i in 1 : p) {
    for (j in 1 : p) {
      cov_mtx[i, j] = 0.3^abs(i - j)
    }
  }
  X = rmvnorm(n, sigma = cov_mtx)
  
  f0 = 2 * X[,1] * X[,4] + 2 * X[,7] * X[,10]
  X = as.data.frame(X)
  dimnames(X)[[2]] = sapply(1 : p, function(s) paste("X", s, sep = ""))
  
  if (binary) {
    f0 = scale(f0, center = T, scale = F)
    prob = pnorm(f0, mean = 0, sd = 1)
    Y = c()
    for (i in 1:n) {
      Y[i] = rbinom(1, 1, prob = prob[i])
    }
    return(list(X = X, Y = Y, f0 = f0, prob = prob))
    
  } else {
    Y = rnorm(n, mean = f0, sd = sigma)
    return(list(X = X, Y = Y, f0 = f0, sigma = sigma))
  }
  
}


#' Generate data with independent and mixed-type predictors
#' 
#' Generate data including responses and predictors values, of which predictors are independent and of mixed types.
#' 
#' Sample the predictors \eqn{x_1, ..., x_{ceiling(p/2)}} from Bernoulli(0.5) independently and
#' \eqn{x_{ceiling(p/2)+1}, ..., x_p} from Uniform(0, 1) independently.
#' If \code{binary = FALSE}, sample the continuous response \eqn{y} from Normal(\eqn{f0(x), \sigma^2}), where 
#' \deqn{f0(x) = 10sin(\pi x_{ceiling(p/2)+1}*x_{ceiling(p/2)+2}) + 20(x_{ceiling(p/2)+3}-0.5)^2 + 10x_1 + 5x_2.}
#' If \code{binary = TRUE}, sample the binary response \eqn{y} from Bernoulli(\eqn{\Phi(f0(x))}) where \eqn{f0} is defined above and
#' \eqn{\Phi} is the cumulative density function of the standard normal distribution.
#' 
#' @param n The number of observations.
#' @param p The number of predictors.
#' @param sigma The error variance.
#' @param binary A boolean argument: \code{binary = TRUE} indicates that binary responses are generated and \code{binary = FALSE} 
#' indicates that continuous responses are generated.
#' @return Return a list with the following components.
#' \item{X}{An n by p data frame representing predictors values, with each row corresponding an observation.}
#' \item{Y}{A vector of length n representing response values.}
#' \item{f0}{A vector of length n representing the values of \eqn{f0(x)}.}
#' \item{sigma}{The error variance which is only returned when \code{binary = FALSE}.}
#' \item{prob}{A vector of length n representing the values of \eqn{\Phi(f0(x))}, which is only returned when \code{binary = TRUE}.}
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#' @examples
#' data = mixone(100, 10, 1, FALSE)
mixone = function(n, p, sigma, binary) {
  k = ceiling(p / 2)
  X = matrix(NA, nrow = n, ncol = p)
  X[, 1:k] = matrix(rbinom(n*k, size = 1, prob = 0.5), nrow = n, ncol = k)
  X[, (k+1):p] = matrix(runif(n*k), nrow = n, ncol = k)
  
  f0 = 10 * sin(pi * X[, k+1] * X[, k+2]) + 20 * (X[, k+3] - 0.5)^2 + 10 * X[, 1] + 5 * X[, 2]
  X = as.data.frame(X)
  for (j in 1:k) {
    X[, j] = factor(X[, j], levels = c("0", "1"), labels = c("0", "1"))
  }
  dimnames(X)[[2]] = sapply(1 : p, function(s) paste("X", s, sep = ""))
  
  if (binary) {
    f0 = scale(f0, center = T, scale = F)
    prob = pnorm(f0, mean = 0, sd = 1)
    Y = c()
    for (i in 1:n) {
      Y[i] = rbinom(1, 1, prob = prob[i])
    }
    return(list(X = X, Y = Y, f0 = f0, prob = prob))
    
  } else {
    Y = rnorm(n, mean = f0, sd = sigma)
    return(list(X = X, Y = Y, f0 = f0, sigma = sigma))
  }
  
}

#' Generate data with correlated and mixed-type predictors
#' 
#' Generate data including responses and predictors values, of which predictors are correlated and of mixed types.
#' 
#' Sample the predictors \eqn{x_1, ..., x_{20}} from Bernoulli(0.2) independently,
#' \eqn{x_{21}, ..., x_{40}} from Bernoulli(0.5) independently,
#' and \eqn{x_{41}, ..., x_{84}} from a multivariate normal distribution with mean 0, variance 1 and correlation 0.3.
#' If \code{binary = FALSE}, sample the continuous response \eqn{y} from Normal(\eqn{f0(x), \sigma^2}), where 
#' \deqn{f0(x) = -4 + x_1 + sin(\pi x_1*x_{44}) - x_{21} + 0.6x_{41}*x_{42} - exp[-2(x_{42}+1)^2] - x_{43}^2 + 0.5x_{44}.}
#' If \code{binary = TRUE}, sample the binary response \eqn{y} from Bernoulli(\eqn{\Phi(f0(x))}) where \eqn{f0} is defined above and 
#' \eqn{\Phi} is the cumulative density function of the standard normal distribution.
#' 
#' @param n The number of observations.
#' @param sigma The error variance.
#' @param binary A boolean argument: \code{binary = TRUE} indicates that binary responses are generated and \code{binary = FALSE} 
#' indicates that continuous responses are generated.
#' @return Return a list with the following components.
#' \item{X}{An n by p data frame representing predictors values, with each row corresponding an observation.}
#' \item{Y}{A vector of length n representing response values.}
#' \item{f0}{A vector of length n representing the values of \eqn{f0(x)}.}
#' \item{sigma}{The error variance which is only returned when \code{binary = FALSE}.}
#' \item{prob}{A vector of length n representing the values of \eqn{\Phi(f0(x))}, which is only returned when \code{binary = TRUE}.}
#' @author Chuji Luo: \email{cjluo@ufl.edu} and Michael J. Daniels: \email{daniels@ufl.edu}.
#' @references 
#' Luo, C. and Daniels, M. J. (2021)
#'   "Variable Selection Using Bayesian Additive Regression Trees."
#'   \emph{arXiv preprint arXiv:2112.13998}.
#' @examples
#' data = mixtwo(100, 1, FALSE)
mixtwo = function(n, sigma, binary) {
  X = matrix(NA, nrow = n, ncol = 84)
  X[, 1:20] = matrix(rbinom(n*20, size = 1, prob = 0.2), nrow = n, ncol = 20)
  X[, 21:40] = matrix(rbinom(n*20, size = 1, prob = 0.5), nrow = n, ncol = 20)
  covmat = matrix(rep(0.3, 1936), nrow = 44, ncol = 44) + diag(x = 0.7, nrow = 44)
  X[, 41:84] = rmvnorm(n, mean = rep(0, 44), sigma = covmat)
  
  f0 = -4 + X[, 1] + sin(pi * X[, 1] * X[, 44]) - X[, 21] + 
    0.6 * X[, 41] * X[, 42] - exp(-2 * (X[, 42] + 1)^2) - X[, 43]^2 + 
    0.5 * X[, 44]
  X = as.data.frame(X)
  for (j in 1:40) {
    X[, j] = factor(X[, j], levels = c("0", "1"), labels = c("0", "1"))
  }
  dimnames(X)[[2]] = sapply(1:84, function(s) paste("X", s, sep=""))
  
  if (binary) {
    f0 = scale(f0, center = T, scale = F)
    prob = pnorm(f0, mean = 0, sd = 1)
    Y = c()
    for (i in 1:n) {
      Y[i] = rbinom(1, 1, prob = prob[i])
    }
    return(list(X = X, Y = Y, f0 = f0, prob = prob))
    
  } else {
    Y = rnorm(n, mean = f0, sd = sigma)
    return(list(X = X, Y = Y, f0 = f0, sigma = sigma))
  }
  
}