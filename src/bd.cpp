/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/bd.cpp.
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
#include "bd.h"

bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma, 
        std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
        std::vector<std::vector<double>>& mr_vec)
{
  tree::npv goodbots;  //nodes we could birth at (split on)
  double PBx = getpb(x,xi,pi,goodbots); //prob of a birth at x (CL add: just proposal)
  
  if(gen.uniform() < PBx) { //do birth or death
    
    //--------------------------------------------------
    //draw proposal
    tree::tree_p nx; //bottom node to split on
    size_t v,c; //splitting variable and cutpoint
    double pr; //part of metropolis ratio from proposal and prior
    bprop(x,xi,pi,goodbots,PBx,nx,v,c,pr,nv,pv,aug,gen);
    
    //--------------------------------------------------
    //compute sufficient statistics
    size_t nr,nl; //counts in proposed bots
    double syl, syr; //sum of y in proposed bots
    getsuff(x,nx,v,c,xi,di,nl,syl,nr,syr);
    
    //--------------------------------------------------
    //compute alpha
    double alpha=0.0, lalpha=0.0, lalpha0=0.0;
    double lhl, lhr, lht;
    if((nl>=5) && (nr>=5)) { //cludge?
      lhl = lh(nl,syl,sigma,pi.tau);
      lhr = lh(nr,syr,sigma,pi.tau);
      lht = lh(nl+nr,syl+syr,sigma,pi.tau);
      
      alpha=1.0;
      lalpha = log(pr) + (lhl+lhr-lht) + log(sigma);  //except log(pr), lalpha is liklihood ratio
      //lalpha0 = log(pr) + (lhl+lhr-lht) + log(sigma);
      lalpha = std::min(0.0,lalpha);
    }
    
    //--------------------------------------------------
    //try metrop
    double mul,mur; //means for new bottom nodes, left and right
    double temp_mr; //temp Metropolis ratio for the potential newly added spliting variable
    //double temp_mr0;
    
    double uu = gen.uniform();
    bool dostep = (alpha > 0) && (log(uu) < lalpha);
    if(dostep) {
      mul = drawnodemu(nl,syl,pi.tau,sigma,gen);
      mur = drawnodemu(nr,syr,pi.tau,sigma,gen);
      
      temp_mr = exp(lalpha);
      //temp_mr0 = exp(lalpha0);
      
      x.birthp(nx,v,c,mul,mur,temp_mr);  //CL add: x is the current tree, nx is the splitting node
      
      //update importance kernels
      nv[v]++;
      mr_vec[v].push_back(temp_mr);
      //mr0_vec[v].push_back(temp_mr0);
      
      return true;
    } else {
      return false;
    }
  } else {
    //--------------------------------------------------
    //draw proposal
    double pr;  //part of metropolis ratio from proposal and prior
    tree::tree_p nx; //nog node to death at
    dprop(x,xi,pi,goodbots,PBx,nx,pr,gen);
    
    //--------------------------------------------------
    //compute sufficient statistics
    size_t nr,nl; //counts at bots of nx
    double syl, syr; //sum at bots of nx
    getsuff(x, nx->getl(), nx->getr(), xi, di, nl, syl, nr, syr);
    
    //--------------------------------------------------
    //compute alpha
    double lhl, lhr, lht;
    lhl = lh(nl,syl,sigma,pi.tau);
    lhr = lh(nr,syr,sigma,pi.tau);
    lht = lh(nl+nr,syl+syr,sigma,pi.tau);
    
    double lalpha = log(pr) + (lht - lhl - lhr) - log(sigma);
    lalpha = std::min(0.0,lalpha);
    
    //--------------------------------------------------
    //try metrop
    double mu;
    double temp_mr;
    //double temp_mr0;
    
    if(log(gen.uniform()) < lalpha) {
      mu = drawnodemu(nl+nr,syl+syr,pi.tau,sigma,gen);
      
      size_t nx_getv = nx->getv();   //split var in proposed node
      
      nv[nx->getv()]--;
      
      x.deathp(nx,mu,temp_mr);
      
      //remove the Metropolis ratio
      auto mr_it = find(mr_vec[nx_getv].begin(), mr_vec[nx_getv].end(), temp_mr);
      int mr_index = distance(mr_vec[nx_getv].begin(), mr_it);
      mr_vec[nx_getv].erase(mr_vec[nx_getv].begin() + mr_index);
      
      //auto mr0_it = find(mr0_vec[nx_getv].begin(), mr0_vec[nx_getv].end(), temp_mr0);
      //int mr0_index = distance(mr0_vec[nx_getv].begin(), mr0_it);
      //mr0_vec[nx_getv].erase(mr0_vec[nx_getv].begin() + mr0_index);
      
      return true;
    } else {
      return false;
    }
  }
}
