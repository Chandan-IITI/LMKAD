% Reference
%   gonen08icml
%   Localized Multiple Kernel Learning
%   Mehmet Gonen, Ethem Alpaydin
% Mehmet Gonen (gonen@boun.edu.tr)
% Department of Computer Engineering, Bogazici University
%   Proceedings of the 25th International Conference on Machine Learning, 2008

function eta = etas(X, gat, eps, typ)
    N = size(X, 1); 
    switch typ
        case {'constant_softmax'}
            val = [ones(N, 1), zeros(size(X))] * gat';
            eta = exp(bsxfun(@minus, val, max(val, [], 2)));
            eta = bsxfun(@rdivide, eta, sum(eta, 2));
        case {'constant_sigmoid'}
            val = -[ones(N, 1), zeros(size(X))] * gat';
            eta = 1 ./ (1 + exp(val));
        case {'linear_sigmoid', 'sigmoid'}
            val = -[ones(N, 1), X] * gat';
            eta = 1 ./ (1 + exp(val));
        case {'linear_softmax', 'softmax'}
            val = [ones(N, 1), X] * gat';
            eta = exp(bsxfun(@minus, val, max(val, [], 2)));
            eta = bsxfun(@rdivide, eta, sum(eta, 2));
        case 'rbf_softmax'
            P = size(gat, 1);
            val = zeros(N, P);
            for m = 1:P
                val(:, m) = -sum(bsxfun(@minus, X, gat(m, 2:end)).^2, 2) / gat(m, 1)^2;
            end
            eta = exp(bsxfun(@minus, val, max(val, [], 2)));
            eta = bsxfun(@rdivide, eta, sum(eta, 2));
    end
    eta(eta < eps) = 0;
    eta(eta > 1 - eps) = 1;
end
