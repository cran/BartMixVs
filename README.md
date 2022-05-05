# BartMixVs
## Overview
This R package is built upon CRAN R package 'BART' version 2.7 (https://github.com/cran/BART) and implements the existing BART-based variable selection approaches discussed in the paper: Luo, C. and Daniels, M. J. (2021). "Variable selection using Bayesian additive regression trees." _arXiv preprint arXiv:2112.13998_, https://doi.org/10.48550/arXiv.2112.13998.

## Background
Bayesian Additive Regression Trees (BART) is a nonparametric regression model which is flexible enough to capture the interactions between predictors and the nonlinear relationships with the response. BART can be used not only for estimation, but also for variable selection. Existing variable selection approaches for BART include:
1. the permutation-based variable selection approach using BART variable inclusion proportions (VIP) as the variable importance, proposed in Bleich, Justin et al. (2014). "Variable selection for BART: an application to gene regulation." _Ann. Appl. Stat._ 8.3, pp 1750-1781;
2. the median probability model from DART which is a variant of BART and proposed in Linero, A. R. (2018). "Bayesian regression trees for high-dimensional prediction and variable selection." _J. Amer. Statist. Assoc._ **113** 626--636;
3. the variable selection approach using ABC Bayesian forests proposed in Liu, Yi, Veronika Rockova, and Yuexi Wang (2021). "Variable selection with ABC Bayesian forests." _J. R. Stat. Soc. Ser. B. Stat. Methodol._ 83.3, pp. 453--481.

Luo and Daniels (2021) review these methods with an emphasis on the capability of identifying relevant predictors. Furthermore, out of the consideration of the existence of mixed-type predictors and the goal of allowing more relevant predictors into the model, Luo and Daniels (2021) propose three new methods for BART:
1. the permutation-based variable selection approach using BART within-type VIP as the variable importance;
2. the permutation-based variable selection approach using BART Metropolis importance (MI) as the variable importance;
3. the backward selection with two filters.

See Luo and Daniels (2021) for more details.

## Philosophy of BartMixVs
The 'BartMixVs' R package provides data sampling functions to generate the simulation data used in Luo and Daniels (2021), inherits estimation functions from the 'BART' package and implements existing BART-based variable selection approaches (3 old + 3 new) previously mentioned.
1. Data sampling functions include `friedman()`, `checkerboard()`, `mixone()` and `mixtwo()` corresponding to Scenario C.C.1 (or B.C.1), Scenario C.C.2 (or B.C.2), Scenario C.M.1 (or B.M.1) and Scenario C.M.2 (or B.M.2) in Luo and Daniels (2021), respectively.
2. BART estimation and prediction functions inherited from the 'BART' package include `wbart()` (`mc.wbart()`), `pbart()` (`mc.pbart()`) and `pwbart()` (`mc.pwbart()`). Note that while most of the original features of the 'BART' functions are kept, two modifications are made:
   - 'BartMixVs' provides two types of split probability for the tree prior in BART:
     - One type is the split probability used in 'BART' and proposed in Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). "BART: Bayesian additive regression trees." _Ann. Appl. Stat._ **4** 266--298:
       <p align="center">
        <img src="https://render.githubusercontent.com/render/math?math=p(d) = \frac{\gamma}{(1 %2B d)^{\beta}},\quad\quad d \, \text{is the depth of the to-be-split node}, \gamma \in (0,1) \, \text{and} \, \beta \in (0, \infty).">
       </p>
     - The other type is the split probability proposed in Rockova V, Saha E (2019). “On theory for BART.” _In The 22nd International Conference on Artificial Intelligence and Statistics_ (pp. 2839–2848). PMLR, which is proved to achieve the optimal posterior contraction:
       <p align="center">
        <img src="https://render.githubusercontent.com/render/math?math=p(d) = \gamma^d,\quad\quad d \, \text{is the depth of the to-be-split node and}\, \gamma \in [1/n,1/2).">
       </p>
   - The second modification is that in addition to variable inclusion proportions, marginal posterior variable inclusion probabilities and posterior split probabilities, the estimation functions in 'BartMixVs' also provide two new variable importance measures: within-type variable importance proportions and Metropolis importance.
3. 'BartMixVs' provides four main functions for the existing BART-based variable selection approaches:
   - The function `permute.vs()` (or `mc.permute.vs()` with parallel computation) implements the permutation-based variable selection approach with three types of variable importance measures being considered: BART variable inclusion proportions (Bleich et al. (2014)), BART within-type variable inclusion proportions and BART Metropolis importance (Luo and Daniels (2021)).
   - The function `mc.backward.vs()` implements the backward selection proposed in Luo and Daniels (2021).
   - The function `medianInclusion.vs()` implements the DART variable selection approach proposed in Linero (2018).
   - The function `abc.vs()` (or `mc.abc.vs()` with parallel computation) implements the ABC Bayesian forests variable selection approach proposed in Liu et al. (2021).
