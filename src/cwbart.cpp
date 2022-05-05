/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/cwbart.cpp.
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
#include "heterbart.h"


#define TRDRAW(a, b) trdraw(a, b)
#define TEDRAW(a, b) tedraw(a, b)


RcppExport SEXP cwbart(
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
    SEXP _itau,
    SEXP _inu,
    SEXP _ilambda,
    SEXP _isigest,
    SEXP _iw,
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
    SEXP _inkeeptestme,
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
  Rcpp::NumericVector  yv(_iy); 
  double *iy = &yv[0];
  Rcpp::NumericVector  xpv(_ixp);
  double *ixp = &xpv[0];
  size_t m = Rcpp::as<int>(_im);
  Rcpp::IntegerVector _nc(_inc);
  int *numcut = &_nc[0];
  size_t nd = Rcpp::as<int>(_ind);
  size_t burn = Rcpp::as<int>(_iburn);
  double mybeta = Rcpp::as<double>(_ipower);
  double alpha = Rcpp::as<double>(_ibase);
  double tau = Rcpp::as<double>(_itau);
  double nu = Rcpp::as<double>(_inu);
  double lambda = Rcpp::as<double>(_ilambda);
  double sigma=Rcpp::as<double>(_isigest);
  Rcpp::NumericVector  wv(_iw); 
  double *iw = &wv[0];
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
  size_t nkeeptestme = Rcpp::as<int>(_inkeeptestme);
  size_t nkeeptreedraws = Rcpp::as<int>(_inkeeptreedraws);
  size_t printevery = Rcpp::as<int>(_inprintevery);
  Rcpp::NumericMatrix Xinfo(_Xinfo);
  bool verbose = Rcpp::as<bool>(_iverbose);
  
  //return data structures (using Rcpp)
  Rcpp::NumericVector trmean(n); //train
  Rcpp::NumericVector temean(np);
  Rcpp::NumericVector sdraw(nd+burn);
  Rcpp::NumericMatrix trdraw(nkeeptrain,n);
  Rcpp::NumericMatrix tedraw(nkeeptest,np);
  
  Rcpp::NumericMatrix varprb(nkeeptreedraws,p);
  Rcpp::IntegerMatrix varcnt(nkeeptreedraws,p);
  //mr_vecs: a list of Metropolis ratios
  //each element of the list is a vector of vector of length p
  //each sub-vector contains the (birth) Metropolis ratios for splits using the predictor at the MCMC sample
  //note: for each sub-vector, first element is 0.0 which is not Metropolis ratio (set for initialization)
  Rcpp::List mr_vecs(nkeeptreedraws);
  //Rcpp::List mr0_vecs(nkeeptreedraws);  // untruncated Metropolis ratios: has the same structure as mr_vecs
  
  
  //random number generation
  arn gen;
  
  //m is the number of trees, bart(size_t m)
  heterbart bm(m);
  
  if(Xinfo.size()>0) {
    xinfo _xi;
    _xi.resize(p);
    for(size_t i=0;i<p;i++) {
      _xi[i].resize(numcut[i]);
      for(size_t j=0;j<numcut[i];j++) _xi[i][j]=Xinfo(i, j);
    }
    bm.setxinfo(_xi);
  }
  
  for(size_t i=0;i<n;i++) trmean[i]=0.0;
  for(size_t i=0;i<np;i++) temean[i]=0.0;
  
  if(verbose) printf("*****Into main of wbart\n");
  //-----------------------------------------------------------
  size_t skiptr,skipte,skipteme,skiptreedraws;
  if(nkeeptrain) {skiptr=nd/nkeeptrain;}
  else skiptr = nd+1;
  if(nkeeptest) {skipte=nd/nkeeptest;}
  else skipte=nd+1;
  if(nkeeptestme) {skipteme=nd/nkeeptestme;}
  else skipteme=nd+1;
  if(nkeeptreedraws) {skiptreedraws = nd/nkeeptreedraws;}
  else skiptreedraws=nd+1;
  
  //--------------------------------------------------
  //print args
  if(verbose) {
    Rcpp::Rcout << "*****Data: " << "n, p, np: " << n << ", " << p << ", " << np << std::endl;
    printf("*****Number of Trees: %zu\n",m);
    if(mybeta > 0)
      Rcpp::Rcout << "*****Prior: split.prob, mybeta, alpha, tau, nu, lambda: " 
                  << "polynomial, " << mybeta << ", " << alpha << ", " << tau << ", " << nu << ", " << lambda << std::endl;
    else
      Rcpp::Rcout << "*****Prior: split.prob, alpha, tau, nu, lambda: " 
                  << "exponential, " << alpha << ", " << tau << ", " << nu << ", " << lambda << std::endl;
      printf("*****sigma: %lf\n", sigma);
      printf("*****w (weights): %lf ... %lf\n", iw[0], iw[n-1]);
      Rcpp::Rcout << "*****Dirichlet: sparse, theta, omega, a, b, rho, augment: " 
                  << dart << ", " << theta << ", " << omega << ", " << a << ", "
                  << b << ", " << rho << ", " << aug << std::endl;
      Rcpp::Rcout << "*****MCMC: (train) nskip, ndpost, keepevery: " << burn << ", " << nkeeptrain << ", " << skiptr << "\n" 
                  << "           (test) nskip, ndpost, keepevery: " << burn << ", " << nkeeptest << ", " << skipte << std::endl;
  }
  
  //--------------------------------------------------
  //heterbart bm(m);
  bm.setprior(alpha,mybeta,tau);
  bm.setdata(p,n,ix,iy,numcut);
  bm.setdart(a,b,rho,aug,dart,theta,omega);
  
  //--------------------------------------------------
  double *svec = new double[n];
  for(size_t i=0;i<n;i++) svec[i]=iw[i]*sigma;
  
  //--------------------------------------------------
  
  std::stringstream treess;  //string stream to write trees to  
  treess.precision(10);
  treess << nkeeptreedraws << " " << m << " " << p << std::endl;
  // dart iterations
  
  std::vector<double> ivarprb (p,0.);
  std::vector<size_t> ivarcnt (p,0);
  
  //--------------------------------------------------
  //temporary storage
  //out of sample fit
  double* fhattest=0; //posterior mean for prediction
  if(np) { fhattest = new double[np]; }
  double restemp=0.0,rss=0.0;
  
  //--------------------------------------------------
  //mcmc
  size_t trcnt=0; //count kept train draws
  size_t tecnt=0; //count kept test draws
  size_t temecnt=0; //count test draws into posterior mean
  size_t treedrawscnt=0; //count kept bart draws
  bool keeptest,keeptestme,keeptreedraw;
  
  time_t tp;
  int time1 = time(&tp);
  xinfo& xi = bm.getxinfo();
  
  for(size_t i=0;i<(nd+burn);i++) {
    
    if(verbose && (i%printevery==0))
      Rcpp::Rcout << "-------BART fit " << i << " out of " << (nd+burn) << std::endl;
    if(i==(burn/2)&&dart) bm.startdart();
    //draw bart (including tree structure, mu's and fitting)
    bm.draw(svec,gen);
    
    //draw sigma
    rss=0.0;
    for(size_t k=0;k<n;k++) {restemp=(iy[k]-bm.f(k))/(iw[k]); rss += restemp*restemp;}
    sigma = sqrt((nu*lambda + rss)/gen.chi_square(n+nu));
    for(size_t k=0;k<n;k++) svec[k]=iw[k]*sigma;
    sdraw[i]=sigma;
    if(i>=burn) {
      for(size_t k=0;k<n;k++) trmean[k]+=bm.f(k);
      if(nkeeptrain && (((i-burn+1) % skiptr) ==0)) {
        for(size_t k=0;k<n;k++) TRDRAW(trcnt,k)=bm.f(k);
        trcnt+=1;
      }
      keeptest = nkeeptest && (((i-burn+1) % skipte) ==0) && np;
      keeptestme = nkeeptestme && (((i-burn+1) % skipteme) ==0) && np;
      if(keeptest || keeptestme) bm.predict(p,np,ixp,fhattest);
      if(keeptest) {
        for(size_t k=0;k<np;k++) TEDRAW(tecnt,k)=fhattest[k];
        tecnt+=1;
      }
      if(keeptestme) {
        for(size_t k=0;k<np;k++) temean[k]+=fhattest[k];
        temecnt+=1;
      }
      keeptreedraw = nkeeptreedraws && (((i-burn+1) % skiptreedraws) ==0);
      if(keeptreedraw) {
        for(size_t j=0;j<m;j++) {
          treess << bm.gettree(j);
        }
        
        ivarcnt=bm.getnv();
        ivarprb=bm.getpv();
        
        size_t k=(i-burn)/skiptreedraws;
        
        mr_vecs[k]=bm.getmrvec();
        //mr0_vecs[k]=bm.getmr0vec();
        
        for(size_t j=0;j<p;j++){
          varcnt(k,j)=ivarcnt[j];
          varprb(k,j)=ivarprb[j];
        }
        
        treedrawscnt +=1;
      }
    }
  }
  
  int time2 = time(&tp);
  if(verbose) printf("Time elapsed: %ds\n",time2-time1);
  for(size_t k=0;k<n;k++) trmean[k]/=nd;
  for(size_t k=0;k<np;k++) temean[k]/=temecnt;
  if(verbose) Rcpp::Rcout << "BART Finished!" << std::endl;
  
  //--------------------------------------------------
  if(fhattest) delete[] fhattest;
  if(svec) delete [] svec;
  
  //--------------------------------------------------
  //return
  
  
  Rcpp::List ret;
  ret["sigma"]=sdraw;
  ret["yhat.train.mean"]=trmean;
  ret["yhat.train"]=trdraw;
  ret["yhat.test.mean"]=temean;
  ret["yhat.test"]=tedraw;
  ret["varcount"]=varcnt;
  ret["varprob"]=varprb;
  ret["mr_vecs"]=mr_vecs;
  //ret["mr0_vecs"]=mr0_vecs;
  
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
