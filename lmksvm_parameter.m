% Localized Multiple Kernel Learning for Anomaly Detection (LMKAD)

% Summary
%   creates a default parameter set for LMKAD

% Output(s)
%   par: constructed parameter set

%%% Author: Chandan Gautam, Ramesh Balaji, K. Sudharsan 
%%%% We modified the code written by Mehmet Gonen (gonen@boun.edu.tr, Department of Computer Engineering, Bogazici University)
%%%% for one-class classification task.

function par = lmksvm_parameter()
    par.C = 1; % regularization parameter
    par.eps = 1e-3; % threshold parameter
    par.gat.typ = 'linear_sigmoid'; % gating model function [linear_softmax, linear_sigmoid, rbf_softmax]
    par.ker = {'l', 'g0.1'}; % kernel functions [l: linear, p:polynomial, g:gaussian]
    par.loc.typ = 'linear'; % gating model complexity [linear, quadratic]
    par.nor.dat = {'true', 'true'}; % if true, apply z-normalization to data
    par.nor.ker = {'true', 'true'}; % if true, make kernel unit diagonal
    par.nor.loc = 'true'; % if true, apply z-normalization to gating model data
    par.opt = 'libsvm'; % optimizer [libsvm, mosek, quadprog, smo] %% Currently only support libsvm. In future release, we will enable this for some more solvers like mosek, quadprog etc.
    par.see = 7332; % seed
    par.tau = 1e-3; % tau parameter for SMO algorithm
end