% Localized Multiple Kernel Learning for Anomaly Detection (LMKAD)

% Summary
%   tests LMKAD on test data with given model

% Input(s)
%   tes: test data
%   mod: LMKAD model

% Output(s)
%   out: classification outputs
%%% Author: Chandan Gautam, Ramesh Balaji, K. Sudharsan 
%%%% We modified the code written by Mehmet Gonen (gonen@boun.edu.tr, Department of Computer Engineering, Bogazici University)
%%%% for one-class classification task.
 
function [out,K] = lmksvm_test(tes, mod, model2)
    P = length(tes) - 1;
    for m = 1:P
        tes{m}.X = normalize_data(tes{m}.X, mod.nor.dat{m});
    end
    loc = locality(tes{P + 1}.X, mod.par.loc.typ);
    loc = normalize_data(loc, mod.nor.loc);
    out.eta = etas(loc, mod.gat, mod.par.eps, mod.par.gat.typ);
    out.dis = mod.b * ones(size(tes{1}.X, 1), 1);
	
    for m = 1:P
        K = kernel(tes{m}, mod.sup{m}, mod.par.ker{m}, mod.par.nor.ker{m});
        out.dis = out.dis + sum((out.eta(:, m) * (mod.sup{m}.alp .* mod.sup{m}.y .* mod.sup{m}.eta)') .* K, 2);
    end
end