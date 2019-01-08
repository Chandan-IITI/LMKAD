% Reference
%%% Author: Chandan Gautam, Ramesh Balaji, K. Sudharsan 
%%%% We modified the code written by Mehmet Gonen (gonen@boun.edu.tr, Department of Computer Engineering, Bogazici University)
%%%% for one-class classification task.

function [a output] = test_model(test_data,test_labels,model,model2,kernels,c)

%test the LMKAD model


test.X=test_data;


 test_data = cell(1, 4);
 test_data{1} = test;
 test_data{2} = test;
 test_data{3} = test;
 test_data{4} = test;

parameters = lmksvm_parameter();
parameters.C = c;
parameters.ker = kernels;
parameters.nor.dat = {'true', 'true','true'};
parameters.nor.ker = {'true', 'true','true'};

output = lmksvm_test(test_data, model,model2);
testSize = length(test_labels);
a=ones(testSize,1);
a(find(output.dis<0))=2;
end
