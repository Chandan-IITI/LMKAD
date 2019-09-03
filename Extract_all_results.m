
clear all;

load('gpp_lsoft.mat')
% load('gpl_lsoft.mat')

counting = 0;
for dn=1:16
    if (dn==5||dn==6)
        continue;
    end
    
    for cnum=1:2 %%% Class number
        if dn==1 & cnum == 2
            continue
        end
        counting = counting + 1;
        all_res{counting,1} =  res_all{dn, cnum}.dname;
        all_res{counting,2}= res_all{dn,cnum}.testmeanrun(1,7);
    end
end