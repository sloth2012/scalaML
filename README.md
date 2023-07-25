# scalaML

Implement some ML algorithms in scala. 

1.  base bgd, sgd, mbgd to get started.
2.  Frequently-used GD algos： 
    -   SGD with momentum and NAG
    -   AdaGrad
    -   AdaDelta
    -   RMSProp
    -   Adam with AMSGrad
    -   AdaMax
    -   Nadam
    -   FTRL_Proximal(online learning)
    -   SWATS
3.  Quasi-Newton Methods
    -   DFP with golden section method
    -   BFGS with golden section method
    -   CG with Fletcher-Reeves method
    -   L-BFGS (implement with line search method of Wolfe-Powell)
4.  common loss function for binary classification.  
5.  algos in package newml:
    -    redesign ml for multi-classification


**note**: 
1.  the package ml is just tested for binary classification(especially logloss) now. 
