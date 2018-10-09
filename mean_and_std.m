function mas = mean_and_std(X, nor)
    if strcmp(nor, 'true') == 1
        mas.mea = mean(X);
        mas.std = std(X);
        mas.std(mas.std == 0) = 1;
    else
        D = size(X, 2);
        mas.mea = zeros(1, D);
        mas.std = ones(1, D);
    end
end