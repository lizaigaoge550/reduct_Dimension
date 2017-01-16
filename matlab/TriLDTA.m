function [T, Y, P, Sb, Sw ] = TriLDTA( M, wname, geshu , leishu)
%%%�Ҹ���Ҷ�õĻ�������õĻ���ֵ��������룬���±꣨ǰgeshu����
%   M������֡����������
if size(M,2)==120
    M(:,[9,69]) = [];%ɾ������
elseif size(M,2)==360
    M(:,[8*3+(1:3),68*3+(1:3)]) = [];
end
%%Gabor�任
if strcmp(wname,'Gabor')
    %���Զ೤
    hehe = 1:size(M,1);
    miao = gabor(hehe');
    DM = zeros(length(miao),size(M,2));
    %����������
    for i = 1:size(M,2)
        DM(:,i) = gabor(M(:,i));
    end

elseif strcmp(wname,'DFT')
    DM = abs(fft(M,[],1));
    DM = DM(1:floor(size(DM,1)/2),:);
    
else
%%С���任
    l = length(wavedec(M(:,1),4,wname));
    DM = zeros(l,size(M,2));
    for i = 1:size(M,2)
        DM(:,i) = wavedec(M(:,i),4,wname);
    end
    DM = DM(1:floor(size(DM,1)/2),:);%ȡǰһ��
end


means = zeros(size(DM,1),leishu);
vars = zeros(size(DM,1),leishu);
for i = 1:leishu
    means(:,i) = mean(DM(:,1+(i-1)*size(DM,2)/leishu:i*size(DM,2)/leishu),2);
end
m = mean(DM,2);%���������ľ�ֵ
dm = zeros(size(DM,1),1);
for i = 1:leishu
    dm = dm + (means(:,i) - m).^2;
end

for i = 1:leishu
    vars(:,i) = var(DM(:,1+(i-1)*size(DM,2)/leishu:i*size(DM,2)/leishu),0,2);
end

d = dm./sum(vars,2);%����
d(isnan(d)) = 0;%������Щbug
d(abs(dm)<0.000001) = 0;%������Щbug
[Y,P] = sort(d,'descend');%������
T = DM(P,:);
Sb = dm(P);
temp = sum(vars,2);
Sw = temp(P);
%%%%%%%%%%%%%%%%%%%%%%%%%%%ɾ�ظ�
h = [1;find(abs(diff(T(:,1)))>0.0002)+1];%��С��<0.0002�����ظ�
T = T(h(1:geshu),:);%ɾ�����ظ�����
P = P(h(1:geshu));
Y = Y(h(1:geshu));
Sb = Sb(h(1:geshu));
Sw = Sw(h(1:geshu));

end