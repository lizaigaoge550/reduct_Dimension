function [features] = main(M)
for j = 1:size(M,3)
    for i = 1:size(M,2)
        features(:,i,j) = gabor(M(:,i,j));
    end
end
end