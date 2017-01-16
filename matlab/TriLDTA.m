function [T, Y, P, Sb, Sw ] = TriLDTA( M, wname, geshu , leishu)
%%%找傅里叶好的基，输出好的基的值，分类距离，和下标（前geshu个）
%   M中行是帧，列是样本
if size(M,2)==120
    M(:,[9,69]) = [];%删空样本
elseif size(M,2)==360
    M(:,[8*3+(1:3),68*3+(1:3)]) = [];
end
%%Gabor变换
if strcmp(wname,'Gabor')
    %试试多长
    hehe = 1:size(M,1);
    miao = gabor(hehe');
    DM = zeros(length(miao),size(M,2));
    %下面是正题
    for i = 1:size(M,2)
        DM(:,i) = gabor(M(:,i));
    end

elseif strcmp(wname,'DFT')
    DM = abs(fft(M,[],1));
    DM = DM(1:floor(size(DM,1)/2),:);
    
else
%%小波变换
    l = length(wavedec(M(:,1),4,wname));
    DM = zeros(l,size(M,2));
    for i = 1:size(M,2)
        DM(:,i) = wavedec(M(:,i),4,wname);
    end
    DM = DM(1:floor(size(DM,1)/2),:);%取前一半
end


means = zeros(size(DM,1),leishu);
vars = zeros(size(DM,1),leishu);
for i = 1:leishu
    means(:,i) = mean(DM(:,1+(i-1)*size(DM,2)/leishu:i*size(DM,2)/leishu),2);
end
m = mean(DM,2);%所有样本的均值
dm = zeros(size(DM,1),1);
for i = 1:leishu
    dm = dm + (means(:,i) - m).^2;
end

for i = 1:leishu
    vars(:,i) = var(DM(:,1+(i-1)*size(DM,2)/leishu:i*size(DM,2)/leishu),0,2);
end

d = dm./sum(vars,2);%距离
d(isnan(d)) = 0;%除掉这些bug
d(abs(dm)<0.000001) = 0;%除掉这些bug
[Y,P] = sort(d,'descend');%列向量
T = DM(P,:);
Sb = dm(P);
temp = sum(vars,2);
Sw = temp(P);
%%%%%%%%%%%%%%%%%%%%%%%%%%%删重复
h = [1;find(abs(diff(T(:,1)))>0.0002)+1];%差小于<0.0002的算重复
T = T(h(1:geshu),:);%删除了重复的行
P = P(h(1:geshu));
Y = Y(h(1:geshu));
Sb = Sb(h(1:geshu));
Sw = Sw(h(1:geshu));

end