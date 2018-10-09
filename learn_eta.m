function [alp, eta, mod, obj, yyKeta, model2] = learn_eta(tra, par, yyKm, alp, eta, mod, obj, yyKeta,model2)
    sup = find(alp ~= 0);
    gra = eta_gradient_lmk(alp(sup), yyKm(sup, sup, :), eta(sup, :), [ones(size(sup, 1), 1), mod.loc(sup, :)], mod.gat, par.gat.typ);
    srn = sqrt(sum(sum(gra.^2)));
    if srn ~= 0
        gra = gra ./ srn;
    else
        return
    end
    coe = 1;
	oldCoe = 0;
    oldObj = obj;
    while 1
        if coe >= 1
            oldAlp = alp; oldEta = eta; oldMod = mod; oldObj = obj;
        end
        mod.gat = mod.gat - (coe - oldCoe) * gra;
        eta = etas(mod.loc, mod.gat, par.eps, par.gat.typ);
        yyKeta = kernel_eta(yyKm, eta);
		[alp, obj, model2] = solve_svm(tra{1}, par, yyKeta, alp);
        mod.sol = mod.sol + 1;
        if obj < oldObj
            if coe >= 1
                oldCoe = coe;
                coe = coe * 2;
            else
                break;
            end
        else
            if coe > 1
                oldMod.sol = mod.sol;
                alp = oldAlp; eta = oldEta; mod = oldMod; obj = oldObj;                
                yyKeta = kernel_eta(yyKm, eta);
                break;
            else
                oldCoe = coe;
                coe = coe / 2;
                if (coe < par.eps)
                    oldMod.sol = mod.sol;
                    alp = oldAlp; eta = oldEta; mod = oldMod; obj = oldObj;
                    yyKeta = kernel_eta(yyKm, eta);
                    break;
                end
            end
        end
    end
end