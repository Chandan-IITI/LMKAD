function [sigma] = getMeanSigma(M_train)

N = size(M_train,2);  %%% Number of samples/training data

Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));

sigma = sqrt(mean(mean(Dtrain)));

end

