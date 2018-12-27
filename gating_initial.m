% Reference
%   gonen08icml
%   Localized Multiple Kernel Learning
%   Mehmet Gonen, Ethem Alpaydin
% Mehmet Gonen (gonen@boun.edu.tr)
% Department of Computer Engineering, Bogazici University
%   Proceedings of the 25th International Conference on Machine Learning, 2008

function gat = gating_initial(loc, P, typ)
    N = size(loc, 1);
    DG = size(loc, 2);
    switch typ
        case {'constant_sigmoid' 'linear_sigmoid' 'sigmoid'}
            gat = 0.02 * rand(P, DG + 1) - 0.01;
        case {'constant_softmax' 'linear_softmax' 'softmax'}
            gat = 0.02 * rand(P, DG + 1) - 0.01;
        case 'rbf_softmax'
            shu = randperm(N);
            gat = ones(P, DG + 1);
            gat(:, 2:end) = loc(shu(1:P), :);
    end
end
