/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/bart.cpp.
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
#include "bart.h"

//--------------------------------------------------
//constructor
bart::bart():m(200),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false) {}
bart::bart(size_t im):m(im),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false) {}
bart::bart(const bart& ib):m(ib.m),t(m),pi(ib.pi),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),di(),dartOn(false)
{
  this->t = ib.t;
}
bart::~bart()
{
  if(allfit) delete[] allfit;
  if(r) delete[] r;
  if(ftemp) delete[] ftemp;
}

//--------------------------------------------------
//operators
bart& bart::operator=(const bart& rhs)
{
  if(&rhs != this) {
    
    this->t = rhs.t;
    this->m = t.size();
    
    this->pi = rhs.pi;
    
    p=0;n=0;x=0;y=0;
    xi.clear();
    
    if(allfit) {delete[] allfit; allfit=0;}
    if(r) {delete[] r; r=0;}
    if(ftemp) {delete[] ftemp; ftemp=0;}
    
  }
  return *this;
}
//--------------------------------------------------
//get,set
void bart::setm(size_t m)
{
  t.resize(m);
  this->m = t.size();
  
  if(allfit && (xi.size()==p)) predict(p,n,x,allfit);
}

//--------------------------------------------------
void bart::setxinfo(xinfo& _xi)
{
  size_t p=_xi.size();
  xi.resize(p);
  for(size_t i=0;i<p;i++) {
    size_t nc=_xi[i].size();
    xi[i].resize(nc);
    for(size_t j=0;j<nc;j++) xi[i][j] = _xi[i][j];
  }
}
//--------------------------------------------------
void bart::setdata(size_t p, size_t n, double *x, double *y, size_t numcut)
{
  int* nc = new int[p];
  for(size_t i=0; i<p; ++i) nc[i]=numcut;
  this->setdata(p, n, x, y, nc);
  delete [] nc;
}

void bart::setdata(size_t p, size_t n, double *x, double *y, int *nc)
{
  this->p=p; this->n=n; this->x=x; this->y=y;
  if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);
  
  if(allfit) delete[] allfit;
  allfit = new double[n];
  predict(p,n,x,allfit);
  
  if(r) delete[] r;
  r = new double[n];
  
  if(ftemp) delete[] ftemp;
  ftemp = new double[n];
  
  di.n=n; di.p=p; di.x = &x[0]; di.y=r;   //CL add: di.y is a pointer whch is set to r
  for(size_t j=0;j<p;j++){
    nv.push_back(0);
    pv.push_back(1/(double)p);
    //initialize mhv_vec
    mr_vec.push_back({0.0});
    //mr0_vec.push_back({0.0});
  }
}
//--------------------------------------------------
void bart::predict(size_t p, size_t n, double *x, double *fp)
  //uses: m,t,xi
{
  double *fptemp = new double[n];
  
  for(size_t j=0;j<n;j++) fp[j]=0.0;
  for(size_t j=0;j<m;j++) {
    fit(t[j],xi,p,n,x,fptemp);
    for(size_t k=0;k<n;k++) fp[k] += fptemp[k];
  }
  
  delete[] fptemp;
}
//--------------------------------------------------
void bart::draw(double sigma, rn& gen)
{
  for(size_t j=0;j<m;j++) {
    //fit the j-th tree at matrix of xi (in treefuns.cpp)
    //ftemp contains the predicted value given by t[j]
    fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) {
      allfit[k] = allfit[k]-ftemp[k];
      r[k] = y[k]-allfit[k];
    }
    
    bd(t[j],xi,di,pi,sigma,nv,pv,aug,gen,mr_vec);  //CL add: update tree structure
    drmu(t[j],xi,di,pi,sigma,gen);  //CL add: update mu
    fit(t[j],xi,p,n,x,ftemp);  //CL add: fit the j-th tree at matrix of xi, given the newly updated tree
    for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
  }
  if(dartOn) {
    draw_s(nv,lpv,theta,gen);
    draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
    for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
  }
}
//--------------------------------------------------
//public functions
void bart::pr() //print to screen
{
  Rcpp::Rcout << "*****bart object:\n";
  Rcpp::Rcout << "m: " << m << std::endl;
  Rcpp::Rcout << "t[0]:\n " << t[0] << std::endl;
  Rcpp::Rcout << "t[m-1]:\n " << t[m-1] << std::endl;
  Rcpp::Rcout << "prior and mcmc info:\n";
  pi.pr();
  if(dart){
    Rcpp::Rcout << "*****dart prior (On):\n";
    Rcpp::Rcout << "a: " << a << std::endl;
    Rcpp::Rcout << "b: " << b << std::endl;
    Rcpp::Rcout << "rho: " << rho << std::endl;
    Rcpp::Rcout << "augmentation: " << aug << std::endl;
  }
  else Rcpp::Rcout << "*****dart prior (Off):\n";
  if(p) Rcpp::Rcout << "data set: n,p: " << n << ", " << p << std::endl;
  else Rcpp::Rcout << "data not set\n";
}
