
%%%% Preparing datasets for the experiments

clear;
clc;
for data_num = 1:1
path = pwd;
%load_data;
%get_dnames;
dname = 'space_ga';
dnames = [path '\' dname];
 tot_data = load([dnames '.mat']);

data_only = tot_data.data(:,1:end-1);
data_labels = tot_data.data(:,end);
data_label = data_labels;
classes_all = unique(data_label);

for first_class=1:length(classes_all)
    fold_run_pos=[];
    fold_run_neg=[];
pos_class = classes_all(first_class); % Select the class taken as positive class
data_label(data_labels==pos_class) = 1; % Target as 1 and all remaining class as 2
data_label(data_labels~=pos_class) = 2; % Target as 1 and all remaining class as 2
target_data = data_only(data_label==1,:);
target_data = cat(2, target_data, ones(size(target_data,1),1));
outlier_data = data_only(data_label==2,:);
outlier_data = cat(2, outlier_data, ones(size(outlier_data,1),1).*2);
tot_run = 5;  tot_fold=5;
%%%%%%%%%%%%%%%%%%%%%%
%ind_pos_all = xlsread([dname '.xlsx'], 'pos');
%ind_neg_all = xlsread([dname '.xlsx'], 'neg');
%%%%%%%%%%%%%%%%%%%%%%%
readformat=1; %%% For reading or loading the required format to write the results

indtrain_xls = 3;
indmtrain_xls = 9;
indtrain=sprintf('L%i', indtrain_xls); 
indmtrain=sprintf('L%i', indmtrain_xls); 
indtest=sprintf('B%i', indtrain_xls); 
indmtest=sprintf('B%i', indmtrain_xls); 

for run=1:tot_run  %%% Total number of time to rerun the experiment
 %run=1;
%%%%% Indices for cross validation and Save the indices %%%%
%%%% Comment these lines if you are loading indices from excel sheet %%%
 ind_pos = crossvalind('Kfold',target_data(:,end),tot_fold);
 ind_neg = crossvalind('Kfold',outlier_data(:,end),tot_fold);
% 
 fold_run_pos(:,run) =  ind_pos;
 fold_run_neg(:,run) =  ind_neg;
% 
 if run==tot_run
%%%% For passing the dataset name through 'dname' variable
 %xlswrite([dname num2str(first_class) '.xlsx'],fold_run_pos, 'pos');
 %xlswrite([dname num2str(first_class) '.xlsx'],fold_run_neg, 'neg');
 save([dname num2str(tot_fold) num2str(first_class) 'pos'],'fold_run_pos');
 save([dname num2str(tot_fold) num2str(first_class) 'neg'],'fold_run_neg');

 %%%%%%% OR %%%%%%%%%%%
%%%% For some specific dataset
%% xlswrite('Heart_healthy.xlsx',fold_run_pos, 'pos');
%% xlswrite('Heart_healthy.xlsx',fold_run_neg, 'neg');
 end
end
end
end