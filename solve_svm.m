% Localized Multiple Kernel Learning for Anomaly Detection (LMKAD)
%%% Author: Chandan Gautam, Ramesh Balaji, K. Sudharsan 

%%% Summary: Currently only support libsvm. In future release, we will enable this for some more solvers like  mosek, quadprog, smo etc.
%%% However, it can be easily modified by adding cases for different solvers and call it from lmksvm_parameter.m

function [alp, obj, model] = solve_svm(tra, par, yyK, alp)
    N = size(tra.X, 1);
    switch lower(par.opt)
        case 'libsvm'
            alp = zeros(N, 1);
            %%% Train the one-class model
            model = svmtrain(tra.y, [(1:1:N)' (tra.y * tra.y') .* yyK], sprintf('-q -s 2 -t 4  -e %g -n 0.05', par.tau));
            alp(model.SVs) = abs(model.sv_coef);
    end
    alp(alp < par.C * par.eps) = 0;
    alp(alp > par.C * (1 - par.eps)) = par.C;
%%%% Objective of dual formulation of LMKAD
    obj = - 0.5 * alp' * yyK * alp;

end