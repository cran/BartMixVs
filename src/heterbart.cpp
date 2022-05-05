/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/heterbart.cpp.
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
#include "heterbart.h"

//--------------------------------------------------
void heterbart::pr()
{
  Rcpp::Rcout << "+++++heterbart object:\n";
  bart::pr();
}
//--------------------------------------------------
void heterbart::draw(double *sigma, rn& gen)
{
  //cout << "Running heterbart::draw" << std::endl;
  for(size_t j=0;j<m;j++) {
    fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) {
      allfit[k] = allfit[k]-ftemp[k];
      //di.y is a pointer which is set to r in setdata()
      r[k] = y[k]-allfit[k];
    }
    heterbd(t[j],xi,di,pi,sigma,nv,pv,aug,gen,mr_vec);
    heterdrmu(t[j],xi,di,pi,sigma,gen);
    fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
  }
  if(dartOn) {
    draw_s(nv,lpv,theta,gen);
    draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
    for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
  }
}
