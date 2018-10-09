% Localized Multiple Kernel Learning for Anomaly Detection (LMKAD)
%%% Author: Chandan Gautam, Ramesh Balaji, K. Sudharsan 

%%% Summary: You can add more datasets for test


 clearvars -except a b res_all data_num path res_acc res_gmean; 
 if data_num==1
    
    dname = 'iris';
    dnames = [path dname];
end

if data_num==2
     
    dname = 'iono';
    dnames = [path dname];
end

if data_num==3
     
    dname = 'pima';
    dnames = [path dname];
end

if data_num==4
     
    dname = 'bupa';
    dnames = [path dname];
end

if data_num==5
     
    dname = '[]';
    dnames = [path dname];
end    
if data_num==6
     
    dname = '[]';
    dnames = [path dname];    
end
if data_num==7
     
    dname = '[]';
    dnames = [path dname];    
end

if data_num==8
     
    dname = 'germ';
    dnames = [path dname];    
end
if data_num==9
     
    dname = 'aust';
    dnames = [path dname];    
end

if data_num==10
     
    dname = 'japa';
    dnames = [path dname];    
end
if data_num==11
     
    dname = 'heart';
    dnames = [path dname];    
end
if data_num==12
     
    dname = 'park';
    dnames = [path dname];    
end
if data_num==13
     
    dname = 'abal';
    dnames = [path dname];    
end
if data_num==14
     
    dname = 'spam';
    dnames = [path dname];    
end
if data_num==15
     
    dname = 'wave';
    dnames = [path dname];    
end
if data_num==16
     
    dname = 'space_ga';
    dnames = [path dname];    
end
