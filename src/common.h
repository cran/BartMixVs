/*
 * This file is modified from a source file of the CRAN R package 
 * 'BART': BART/src/common.h.
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
#ifndef GUARD_common_h
#define GUARD_common_h

// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstddef>
#include <typeinfo>
#include <map>
#include <algorithm>

using std::endl;

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Rcpp.h>

#define printf Rprintf

#define cout Rcpp::Rcout

// log(2*pi)
#define LTPI 1.837877066409345483560659472811

// log(pi)
#define logpi 1.1447298858494001741434273513530587

// sqrt(2*pi)
#define RTPI 2.506628274631000502415765284811

#include "rn.h"

#endif
