/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/cpbart.cpp.
 * See below for the copyright of the CRAN R package 'BART'.
 * 
 * BART: Bayesian Additive Regression Trees
 * Copyright (C) 2018 Robert McCulloch and Rodney Sparapani
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, a copy is available at
 * https://www.R-project.org/Licenses/GPL-2
 */
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "rtnorm.h" 


#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)

RcppExport SEXP cpbart(
    SEXP _in,    //number of observations in training data
    SEXP _ip,		//dimension of x
    SEXP _inp,		//number of observations in test data
    SEXP _ix,		//x, train,  pxn (transposed so rows are contiguous in memory)
    SEXP _iy,		//y, train,  nx1
    SEXP _ixp,		//x, test, pxnp (transposed so rows are contiguous in memory)
    SEXP _im,		//number of trees
    SEXP _inc,		//number of cut points
    SEXP _ind,		//number of kept draws (except for thinnning ..)
    SEXP _iburn,		//number of burn-in draws skipped
    SEXP _ipower,
    SEXP _ibase,
    SEXP _binaryOffset,
    SEXP _itau,
    SEXP _idart,
    SEXP _itheta,
    SEXP _iomega,
    SEXP _igrp,
    SEXP _ia,
    SEXP _ib,
    SEXP _irho,
    SEXP _iaug,
    SEXP _inkeeptrain,
    SEXP _inkeeptest,
    SEXP _inkeeptreedraws,
    SEXP _inprintevery,
    SEXP _Xinfo,
    SEXP _iverbose
)
{
  //--------------------------------------------------
  //process args
  size_t n = Rcpp::as<int>(_in);
  size_t p = Rcpp::as<int>(_ip);
  size_t np = Rcpp::as<int>(_inp);
  Rcpp::NumericVector  xv(_ix);
  double *ix = &xv[0];
  Rcpp::IntegerVector  yv(_iy); // binary
  int *iy = &yv[0];
  Rcpp::NumericVector  xpv(_ixp);
  double *ixp = &xpv[0];
  size_t m = Rcpp::as<int>(_im);
  Rcpp::IntegerVector _nc(_inc);
  int *numcut = &_nc[0];
  size_t nd = Rcpp::as<int>(_ind);
  size_t burn = Rcpp::as<int>(_iburn);
  double mybeta = Rcpp::as<double>(_ipower);
  double alpha = Rcpp::as<double>(_ibase);
  double binaryOffset = Rcpp::as<double>(_binaryOffset);
  double tau = Rcpp::as<double>(_itau);
  bool dart;
  if(Rcpp::as<int>(_idart)==1) dart=true;
  else dart=false;
  double a = Rcpp::as<double>(_ia);
  double b = Rcpp::as<double>(_ib);
  double rho = Rcpp::as<double>(_irho);
  bool aug;
  if(Rcpp::as<int>(_iaug)==1) aug=true;
  else aug=false;
  double theta = Rcpp::as<double>(_itheta);
  double omega = Rcpp::as<double>(_iomega);
  Rcpp::IntegerVector _grp(_igrp);
  int *grp = &_grp[0];
  size_t nkeeptrain = Rcpp::as<int>(_inkeeptrain);
  size_t nkeeptest = Rcpp::as<int>(_inkeeptest);
  size_t nkeeptreedraws = Rcpp::as<int>(_inkeeptreedraws);
  size_t printevery = Rcpp::as<int>(_inprintevery);
  Rcpp::NumericMatrix Xinfo(_Xinfo);
  bool verbose = Rcpp::as<bool>(_iverbose);
  
  //return data structures (using Rcpp)
  Rcpp::NumericMatrix trdraw(nkeeptrain,n);
  Rcpp::NumericMatrix tedraw(nkeeptest,np);
  
  
  //variable importance
  Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
  Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
  Rcpp::List mr_vecs(nkeeptreedraws);
  
  
  //random number generation
  arn gen;
  
  bart bm(m);
  
  if(Xinfo.size()>0) {
    xinfo _xi;
    _xi.resize(p);
    for(size_t i=0;i<p;i++) {
      _xi[i].resize(numcut[i]);
      for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
    }
    bm.setxinfo(_xi);
  }
  
  double* iz = new double[n];
  
  std::stringstream treess;  //string stream to write trees to
  treess.precision(10);
  treess << nkeeptreedraws << " " << m << " " << p << std::endl;
  // dart iterations
  std::vector<double> ivarprb (p,0.);
  std::vector<size_t> ivarcnt (p,0);
  
  if(verbose) printf("*****Into main of pbart\n");
  
  size_t skiptr,skipte,skiptreedraws;
  if(nkeeptrain) {skiptr=nd/nkeeptrain;}
  else skiptr = nd+1;
  if(nkeeptest) {skipte=nd/nkeeptest;}
  else skipte=nd+1;
  
  if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
  else skiptreedraws=nd+1;
  
  
  //--------------------------------------------------
  //print args
  if(verbose) {
    Rcpp::Rcout << "*****Data: " << "n, p, np: " << n << ", " << p << ", " << np << std::endl;
    printf("*****BinaryOffset: %lf\n",binaryOffset);
    printf("*****Number of Trees: %zu\n",m);
    if(mybeta > 0)
      Rcpp::Rcout << "*****Prior: split.prob, mybeta, alpha, tau: polynomial, " << mybeta << ", " << alpha << ", " << tau << std::endl;
    else
      Rcpp::Rcout << "*****Prior: split.prob, alpha, tau: exponential, " << alpha << ", " << tau << std::endl;
    Rcpp::Rcout << "*****Dirichlet: sparse, theta, omega, a, b, rho, augment: " 
                << dart << ", " << theta << ", " << omega << ", " << a << ", "
                << b << ", " << rho << ", " << aug << std::endl;
    Rcpp::Rcout << "*****MCMC: (train) nskip, ndpost, keepevery: " << burn << ", " << nkeeptrain << ", " << skiptr << "\n" 
                << "           (test) nskip, ndpost, keepevery: " << burn << ", " << nkeeptest << ", " << skipte << std::endl;
  }
  
  //--------------------------------------------------
  //bart bm(m);
  bm.setprior(alpha,mybeta,tau);
  bm.setdata(p,n,ix,iz,numcut);
  bm.setdart(a,b,rho,aug,dart,theta,omega);
  
  //--------------------------------------------------
  //initialize the latent z's
  for(size_t k=0; k<n; k++) {
    if(iy[k]==0) iz[k]= -rtnorm(0., binaryOffset, 1., gen);
    else iz[k]=rtnorm(0., -binaryOffset, 1., gen);
  }
  
  //--------------------------------------------------
  //temporary storage
  //out of sample fit
  double* fhattest=0; 
  if(np) { fhattest = new double[np]; }
  
  //--------------------------------------------------
  //mcmc
  size_t trcnt=0; //count kept train draws
  size_t tecnt=0; //count kept test draws
  bool keeptest, keeptreedraw;
  
  time_t tp;
  int time1 = time(&tp);
  xinfo& xi = bm.getxinfo();
  
  for(size_t i=0;i<(nd+burn);i++) {
    if(verbose && (i%printevery==0)) 
      Rcpp::Rcout << "-------BART fit " << i << " out of " << (nd+burn) << std::endl;
    
    if(i==(burn/2)&&dart) bm.startdart();
    //draw bart: update tree structures and mu's
    bm.draw(1., gen);
    
    //update latent variable z's
    for(size_t k=0; k<n; k++) {
      if(iy[k]==0) iz[k]= -rtnorm(-bm.f(k), binaryOffset, 1., gen);
      else iz[k]=rtnorm(bm.f(k), -binaryOffset, 1., gen);
    }
    
    if(i>=burn) {
      if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
        for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
        trcnt+=1;
      }
      keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
      if(keeptest) {
        bm.predict(p,np,ixp,fhattest);
        for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
        tecnt+=1;
      }
      keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
      if(keeptreedraw) {
        for(size_t j=0;j<m;j++) {
          treess << bm.gettree(j);
        }
        
        ivarcnt=bm.getnv();
        ivarprb=bm.getpv();
        
        size_t k=(i-burn)/skiptreedraws;
        
        //update variable importance
        mr_vecs[k]=bm.getmrvec();
        
        for(size_t j=0;j<p;j++){
          varcnt(k,j)=ivarcnt[j];
          varprb(k,j)=ivarprb[j];
        }
      }
    }
  }
  
  int time2 = time(&tp);
  if(verbose) printf("Time elapsed: %ds\n",time2-time1);
  if(verbose) Rcpp::Rcout << "BART Finished!" << std::endl;
  
  //--------------------------------------------------
  if(fhattest) delete[] fhattest;
  delete[] iz;
  
  //--------------------------------------------------
  //return
  Rcpp::List ret;
  ret["yhat.train"]=trdraw;
  ret["yhat.test"]=tedraw;
  ret["varcount"]=varcnt;
  ret["varprob"]=varprb;
  ret["mr_vecs"]=mr_vecs;
  
  Rcpp::List xiret(xi.size());
  for(size_t i=0;i<xi.size();i++) {
    Rcpp::NumericVector vtemp(xi[i].size());
    std::copy(xi[i].begin(),xi[i].end(),vtemp.begin());
    xiret[i] = Rcpp::NumericVector(vtemp);
  }
  
  Rcpp::List treesL;
  treesL["cutpoints"] = xiret;
  treesL["trees"]=Rcpp::CharacterVector(treess.str());
  ret["treedraws"] = treesL;
  
  return ret;
  
}
