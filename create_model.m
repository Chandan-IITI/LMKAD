% Reference
%   gonen08icml
%   Localized Multiple Kernel Learning
%   Mehmet Gonen, Ethem Alpaydin
% Mehmet Gonen (gonen@boun.edu.tr)
% Department of Computer Engineering, Bogazici University
%   Proceedings of the 25th International Conference on Machine Learning, 2008

function [model,model2]=create_model(train_data,train_labels,kernels,c)

%training data

training.ind = (1:size(train_data,1))';
training.X = train_data;
training.y = train_labels;



training_data = cell(1, 4);           
training_data{1} = binarize(training);
training_data{2} = binarize(training);
training_data{3} = binarize(training);
training_data{4} = binarize(training);

parameters = lmksvm_parameter();
parameters.C = c;

parameters.ker = kernels;
parameters.nor.dat = {'true', 'true','true'};
parameters.nor.ker = {'true', 'true','true'};

[model, model2] = lmksvm_train(training_data, parameters);
end
