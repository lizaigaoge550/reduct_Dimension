clear
%%%识别情绪，取1个人的12段，每段数据开始的相位相同(取右脚z位置极大时-6)
%%%6用6个关节点，7用24关节点
%wname = 'db2';
wname = 'Gabor';
%%%%%%%%%%%%%%读数据
load('ALL1.mat');%filenames = {'SpineBase','AnkleRight','AnkleLeft','KneeLeft','KneeRight','WristLeft','WristRight'};
load('ALL2.mat');
%filenames = {'SpineBase','AnkleRight','AnkleLeft','ElbowLeft','ElbowRight','FootLeft','FootRight','HandLeft','HandRight','HandTipLeft','HandTipRight','Head','HipLeft',...
%    'HipRight','KneeLeft','KneeRight','Neck','ShoulderLeft','ShoulderRight','SpineMid','SpineShoulder','ThumbLeft','ThumbRight','WristLeft','WristRight'};
%%%%%%%%%%%%%%预处理
dMs = cell(1,180);%做差分后的数据
Ms = cell(1,180);%没做差分的数据
%AK1的数据
for r = 1:120
    temp = ALL1{r};
    if ~isempty(temp)
%%%%%%%%%%%%%%以'SpineBase'的数据为坐标原点（在前3列），去除人的位置的影响        
        for i = 2:25
            temp(:,3*i-2) = temp(:,3*i-2)-temp(:,1);
            temp(:,3*i-1) = temp(:,3*i-1)-temp(:,2);  
            temp(:,3*i) = temp(:,3*i)-temp(:,3);              
        end
%%%%%%%%%%%%%%取均值滤波，平滑一下  
        for i = 1:25
            temp(:,3*i-2) = conv(temp(:,3*i-2),[1,4,6,4,1]/16,'same');
            temp(:,3*i-1) = conv(temp(:,3*i-1),[1,4,6,4,1]/16,'same');
            temp(:,3*i) = conv(temp(:,3*i),[1,4,6,4,1]/16,'same');
        end
%%%%%%%%%%%%%%x,y,z做差分，去除身高等因素的影响 
%可能会造成曲线更加不光滑，频率更分散  
        dMs{r} = diff(temp);
        Ms{r} = temp;
    end
end
%AK2的后半数据
for r = 61:120
    temp = ALL2{r};
    if ~isempty(temp)
%%%%%%%%%%%%%%以'SpineBase'的数据为坐标原点（在前3列），去除人的位置的影响        
        for i = 2:25
            temp(:,3*i-2) = temp(:,3*i-2)-temp(:,1);
            temp(:,3*i-1) = temp(:,3*i-1)-temp(:,2);  
            temp(:,3*i) = temp(:,3*i)-temp(:,3);              
        end
%%%%%%%%%%%%%%取均值滤波，平滑一下  
        for i = 1:25
            temp(:,3*i-2) = conv(temp(:,3*i-2),[1,4,6,4,1]/16,'same');
            temp(:,3*i-1) = conv(temp(:,3*i-1),[1,4,6,4,1]/16,'same');
            temp(:,3*i) = conv(temp(:,3*i),[1,4,6,4,1]/16,'same');
        end
%%%%%%%%%%%%%%x,y,z做差分，去除身高等因素的影响 
%可能会造成曲线更加不光滑，频率更分散  
        dMs{r+60} = diff(temp);
        Ms{r+60} = temp;
    end
end

%%%%%%%%%%%%%%切段
pMs = cell(1,180);
MMM = cell(1,180);
for r = 1:180
    temp = dMs{r};
    tempM = Ms{r};
    if ~isempty(temp)
        p = find(temp(:,end)>1000);%最后1列为时间
        p = [0;p;size(temp,1)];
        pieces = cell(1,length(p)-1);
        piecesM = pieces; 
        for i = 1:length(p)-1
            piece = temp(p(i)+1:p(i+1),:);              
            pieces{i} = piece;%第3列为z  
            pieceM = tempM(p(i)+1:p(i+1),:);              
            piecesM{i} = pieceM;%第3列为z              
        end      
        pMs{r} = pieces; 
        MMM{r} = piecesM;
    end
end

%取每人够长的4段，截成一样长。其中3段做训练集选特征，1段做测试集做同样的处理。
newdata = zeros(64,180*4,24*3);
for r = 1:180
    tempdata = MMM{r};
    if ~isempty(tempdata)
    count = 0;%计数多少段
    for i = 1:length(tempdata)
        temp = [];
        if count == 4
            break;%4段够了
        end
        if size(tempdata{i},1)>100
            count = count + 1;
            hehe = tempdata{i};
            Lmax = find(diff(sign(diff(hehe(:,6))))== -2); % logic vector for the local max value
            if size(hehe,1)-Lmax(1)+1 < 64  %补0 
                temp = zeros(64,size(tempdata{i},2));  
                temp(1:size(hehe,1)-Lmax(1)+1,1,:) = hehe(Lmax(1):end,:);
            else
                temp = hehe(Lmax(1):end,:);
            end
            for j = 4:75
                newdata(:,4*(r-1)+count,j-3) = temp(1:64,j);%取出一个人的数据放好
            end            
        end
    end
    end
end
%删空数据
newdata(:,[33:36,273:276,513:516],:) = [];


%选出高兴和愤怒的数据，分成训练集和测试集
testdata = newdata(:,(60:177)*4,:);
traindata = newdata;
traindata(:,(60:177)*4,:) = [];
traindata(:,1:236,:) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%以下是选特征过程
geshu = 9;              %每个坐标选的特征数
leishu = 2;             %类别数
duanshu = 59*3;         %样本数
D = zeros(geshu,24*3);  %特征分类的距离，越大越好
P = zeros(geshu,24*3);  %特征的位置
Tdata = zeros(geshu,leishu*duanshu,24*3);
for i = 1:72
    [Tdata(:,:,i),D(:,i),P(:,i)] = TriLDTA(traindata(:,:,i),wname,geshu,leishu);
end

%%特征重新排列一下
features = zeros(leishu*duanshu,24*3*geshu);
for r = 1:leishu*duanshu
    temp = Tdata(:,r,:);
	features(r,:) = temp(:)';
end    

save('features.mat','features');
temp = repmat(1:2,177,1);
flag = temp(:);
features = [features,flag];
features = [1:size(features,2);features];
% 
csvwrite('ankle_knee_train.csv',features);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%测试集的变换
testfeatures = ceshiji(testdata,wname,P);
save('testfeatures.mat','testfeatures');
temp = repmat(1:2,59,1);
flag = temp(:);
testfeatures = [testfeatures,flag];
testfeatures = [1:size(testfeatures,2);testfeatures];
csvwrite('ankle_knee_test.csv',testfeatures);










    