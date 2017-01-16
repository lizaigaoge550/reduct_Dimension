clear
%%%ʶ��������ȡ1���˵�12�Σ�ÿ�����ݿ�ʼ����λ��ͬ(ȡ�ҽ�zλ�ü���ʱ-6)
%%%6��6���ؽڵ㣬7��24�ؽڵ�
%wname = 'db2';
wname = 'Gabor';
%%%%%%%%%%%%%%������
load('ALL1.mat');%filenames = {'SpineBase','AnkleRight','AnkleLeft','KneeLeft','KneeRight','WristLeft','WristRight'};
load('ALL2.mat');
%filenames = {'SpineBase','AnkleRight','AnkleLeft','ElbowLeft','ElbowRight','FootLeft','FootRight','HandLeft','HandRight','HandTipLeft','HandTipRight','Head','HipLeft',...
%    'HipRight','KneeLeft','KneeRight','Neck','ShoulderLeft','ShoulderRight','SpineMid','SpineShoulder','ThumbLeft','ThumbRight','WristLeft','WristRight'};
%%%%%%%%%%%%%%Ԥ����
dMs = cell(1,180);%����ֺ������
Ms = cell(1,180);%û����ֵ�����
%AK1������
for r = 1:120
    temp = ALL1{r};
    if ~isempty(temp)
%%%%%%%%%%%%%%��'SpineBase'������Ϊ����ԭ�㣨��ǰ3�У���ȥ���˵�λ�õ�Ӱ��        
        for i = 2:25
            temp(:,3*i-2) = temp(:,3*i-2)-temp(:,1);
            temp(:,3*i-1) = temp(:,3*i-1)-temp(:,2);  
            temp(:,3*i) = temp(:,3*i)-temp(:,3);              
        end
%%%%%%%%%%%%%%ȡ��ֵ�˲���ƽ��һ��  
        for i = 1:25
            temp(:,3*i-2) = conv(temp(:,3*i-2),[1,4,6,4,1]/16,'same');
            temp(:,3*i-1) = conv(temp(:,3*i-1),[1,4,6,4,1]/16,'same');
            temp(:,3*i) = conv(temp(:,3*i),[1,4,6,4,1]/16,'same');
        end
%%%%%%%%%%%%%%x,y,z����֣�ȥ����ߵ����ص�Ӱ�� 
%���ܻ�������߸��Ӳ��⻬��Ƶ�ʸ���ɢ  
        dMs{r} = diff(temp);
        Ms{r} = temp;
    end
end
%AK2�ĺ������
for r = 61:120
    temp = ALL2{r};
    if ~isempty(temp)
%%%%%%%%%%%%%%��'SpineBase'������Ϊ����ԭ�㣨��ǰ3�У���ȥ���˵�λ�õ�Ӱ��        
        for i = 2:25
            temp(:,3*i-2) = temp(:,3*i-2)-temp(:,1);
            temp(:,3*i-1) = temp(:,3*i-1)-temp(:,2);  
            temp(:,3*i) = temp(:,3*i)-temp(:,3);              
        end
%%%%%%%%%%%%%%ȡ��ֵ�˲���ƽ��һ��  
        for i = 1:25
            temp(:,3*i-2) = conv(temp(:,3*i-2),[1,4,6,4,1]/16,'same');
            temp(:,3*i-1) = conv(temp(:,3*i-1),[1,4,6,4,1]/16,'same');
            temp(:,3*i) = conv(temp(:,3*i),[1,4,6,4,1]/16,'same');
        end
%%%%%%%%%%%%%%x,y,z����֣�ȥ����ߵ����ص�Ӱ�� 
%���ܻ�������߸��Ӳ��⻬��Ƶ�ʸ���ɢ  
        dMs{r+60} = diff(temp);
        Ms{r+60} = temp;
    end
end

%%%%%%%%%%%%%%�ж�
pMs = cell(1,180);
MMM = cell(1,180);
for r = 1:180
    temp = dMs{r};
    tempM = Ms{r};
    if ~isempty(temp)
        p = find(temp(:,end)>1000);%���1��Ϊʱ��
        p = [0;p;size(temp,1)];
        pieces = cell(1,length(p)-1);
        piecesM = pieces; 
        for i = 1:length(p)-1
            piece = temp(p(i)+1:p(i+1),:);              
            pieces{i} = piece;%��3��Ϊz  
            pieceM = tempM(p(i)+1:p(i+1),:);              
            piecesM{i} = pieceM;%��3��Ϊz              
        end      
        pMs{r} = pieces; 
        MMM{r} = piecesM;
    end
end

%ȡÿ�˹�����4�Σ��س�һ����������3����ѵ����ѡ������1�������Լ���ͬ���Ĵ���
newdata = zeros(64,180*4,24*3);
for r = 1:180
    tempdata = MMM{r};
    if ~isempty(tempdata)
    count = 0;%�������ٶ�
    for i = 1:length(tempdata)
        temp = [];
        if count == 4
            break;%4�ι���
        end
        if size(tempdata{i},1)>100
            count = count + 1;
            hehe = tempdata{i};
            Lmax = find(diff(sign(diff(hehe(:,6))))== -2); % logic vector for the local max value
            if size(hehe,1)-Lmax(1)+1 < 64  %��0 
                temp = zeros(64,size(tempdata{i},2));  
                temp(1:size(hehe,1)-Lmax(1)+1,1,:) = hehe(Lmax(1):end,:);
            else
                temp = hehe(Lmax(1):end,:);
            end
            for j = 4:75
                newdata(:,4*(r-1)+count,j-3) = temp(1:64,j);%ȡ��һ���˵����ݷź�
            end            
        end
    end
    end
end
%ɾ������
newdata(:,[33:36,273:276,513:516],:) = [];


%ѡ�����˺ͷ�ŭ�����ݣ��ֳ�ѵ�����Ͳ��Լ�
testdata = newdata(:,(60:177)*4,:);
traindata = newdata;
traindata(:,(60:177)*4,:) = [];
traindata(:,1:236,:) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%������ѡ��������
geshu = 9;              %ÿ������ѡ��������
leishu = 2;             %�����
duanshu = 59*3;         %������
D = zeros(geshu,24*3);  %��������ľ��룬Խ��Խ��
P = zeros(geshu,24*3);  %������λ��
Tdata = zeros(geshu,leishu*duanshu,24*3);
for i = 1:72
    [Tdata(:,:,i),D(:,i),P(:,i)] = TriLDTA(traindata(:,:,i),wname,geshu,leishu);
end

%%������������һ��
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

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���Լ��ı任
testfeatures = ceshiji(testdata,wname,P);
save('testfeatures.mat','testfeatures');
temp = repmat(1:2,59,1);
flag = temp(:);
testfeatures = [testfeatures,flag];
testfeatures = [1:size(testfeatures,2);testfeatures];
csvwrite('ankle_knee_test.csv',testfeatures);










    