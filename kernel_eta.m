% Reference
%   gonen08icml
%   Localized Multiple Kernel Learning
%   Mehmet Gonen, Ethem Alpaydin
% Mehmet Gonen (gonen@boun.edu.tr)
% Department of Computer Engineering, Bogazici University
%   Proceedings of the 25th International Conference on Machine Learning, 2008

function Keta = kernel_eta(Km, eta)
    N = size(Km, 1);
    P = size(Km, 3);
    Keta = zeros(N, N);
    for m = 1:P
        Keta = Keta + (eta(:, m) * eta(:, m)') .* Km(:, :, m);
    end
end
