function [y] = gabor(x)%%输出x的gabor变换，为长l=128的列向量
y = [];
t = 1:length(x);
for d = 0:log2(length(x))-2
    a = 2^(2*d+2);
    for p = 0:length(x)/(2^d)-1
        b = p*2^d;
        for q = 0:2^d-1
            e = 2*pi*q/(2^d);
            g = exp(-a*(t-b).^2+1i*e*t);
            y = [y;sum(g'.*x)];
        end
    end
end
%y = [real(y);imag(y)];
y = real(y);
%y = abs(y);
end