
%%%% Following script is used from the following reference:

% Localized Multiple Kernel Support Vector Machine (LMKSVM)
% Reference
%   gonen08icml
%   Localized Multiple Kernel Learning
%   Mehmet Gonen, Ethem Alpaydin
%   Proceedings of the 25th International Conference on Machine Learning, 2008

% Summary
%   trains LMKSVM on training data with given parameters

% Input(s)
%   tra: training data
%   par: parameters

% Output(s)
%   mod: LMKSVM model

% Mehmet Gonen (gonen@boun.edu.tr)
% Department of Computer Engineering, Bogazici University

function [mod,model2] = lmksvm_train(tra, par)
    rand('twister', par.see); %#ok<RAND>
    P = length(tra) - 1;
    for m = 1:P
        mod.nor.dat{m} = mean_and_std(tra{m}.X, par.nor.dat{m});
        tra{m}.X = normalize_data(tra{m}.X, mod.nor.dat{m});
    end
    mod.loc = locality(tra{P + 1}.X, par.loc.typ);
    mod.nor.loc = mean_and_std(mod.loc, par.nor.loc);
    mod.loc = normalize_data(mod.loc, mod.nor.loc);
    mod.gat = gating_initial(mod.loc, P, par.gat.typ);
    eta = etas(mod.loc, mod.gat, par.eps, par.gat.typ);
    N = size(tra{1}.X, 1);
    yyKm = zeros(N, N, P);
    for m = 1:P
        yyKm(:, :, m) = (tra{m}.y * tra{m}.y') .* kernel(tra{m}, tra{m}, par.ker{m}, par.nor.ker{m});
    end
    yyKeta = kernel_eta(yyKm, eta);
    alp = zeros(N, 1);
    [alp, obj, model2] = solve_svm(tra{1}, par, yyKeta, alp);
    %display(sprintf('obj1:%10.6f', obj));
    mod.obj = obj;
    mod.sol = 1;
    while 1 && P > 1
        oldObj = obj;
        [alp, eta, mod, obj, yyKeta,model2] = learn_eta(tra, par, yyKm, alp, eta, mod, obj, yyKeta,model2);
        %display(sprintf('obj: %10.6f', obj));
        mod.obj = [mod.obj, obj];
        if abs(obj - oldObj) <= par.eps * abs(oldObj)
            break;
        end
    end
    mod = rmfield(mod, 'loc');
    for m = 1:P
        sup = find(alp .* eta(:, m) ~= 0);
        mod.sup{m}.ind = tra{m}.ind(sup);
        mod.sup{m}.X = tra{m}.X(sup, :);
        mod.sup{m}.y = tra{m}.y(sup);
        mod.sup{m}.alp = alp(sup);
        mod.sup{m}.eta = eta(sup, m);
    end
    sup = find(alp ~= 0);
    act = find(alp ~= 0 & alp < par.C);
    if isempty(act) == 0
        mod.b =  mean(tra{1}.y(act) .* (1 - yyKeta(act, sup) * alp(sup)));
    else
        mod.b = 0;
    end
    mod.par = par;
end

