% SVM classifiers based on  MVD-ELM weighted features demo.
 %%%%%%%%%%%%%%%%%%%%%%%%%% 
clear all
load modelNetTrain128.mat;
TrainNum=3991;
TestNum=908;
% Load training dataset 
P=imageTrainData;
clear imageTrainData
T=trainLabel';
P=P(:,:,:,1:TrainNum); 
T=T(:,1:TrainNum); 
NumberofTrainingData=size(P,4);  
number_class=10;
numLabels=number_class;
NumberofOutputNeurons=number_class;
% Processing the targets of training 
temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if j == T(1,i)
            break; 
        end
    end
    temp_T(j,i)=1;
end
T=temp_T*2-1;    
% Set the size stuff 
AngelNum=6; 
input_size=128;
r_size=2;
r_size_2=2;
pool_size=8;
k_size=1;
k_size_2=1;
pool_size_2=2;
size_map= input_size; 
size_pooled_map =ceil( (size_map)/pool_size);
size_map_2=size_pooled_map;
size_pooled_map_2 =ceil( (size_map_2)/pool_size_2);
InputMask=maskTrainData(:,:,:,1:TrainNum);
clear maskTrainData;
InputMask=InputMask(:,:,1:AngelNum,:);
c_rho=0.01;
N=NumberofTrainingData;
InputDataLayer = P;
clear P;
train_time=tic;
%%%%%%%%%%%%%%%%%%%%%%%%%% 1st layer
%  First step: generate random  convulational kernels
InputWeight = zeros(r_size* r_size, k_size);
for i=1:k_size
    randomWeight_r=rand(r_size,r_size); 
    randomWeight_r = reshape(randomWeight_r, [r_size*r_size 1]);
    InputWeight(:,i)=randomWeight_r ;      
end
A =reshape(InputWeight, [r_size r_size k_size]);
%  Second step: Convulational computation 
X= InputDataLayer(:,:,1:AngelNum,:,:);
clear InputDataLayer;
C=zeros(size_map,size_map,AngelNum,N,k_size);
for k=1:k_size
    C(:,:,:,:,k)= convn(X, A(:,:,k),'same'); 
     C(:,:,:,:,k)= C(:,:,:,:,k)/sum(sum(A(:,:,k)));
    C(:,:,:,:,k) =C(:,:,:,:,k) .*InputMask;
end
clear A1Rep
clear X
clear InputMask
% Third step: pooling
H=zeros(size_pooled_map,size_pooled_map, AngelNum,N,  k_size);
for k=1:k_size
     z=C(:,:,:,:,k);
     HTemp= z(1 : pool_size : end, 1 : pool_size : end, :,:);
     H(:,:,:,:,k)= HTemp;
end
clear C
clear z
clear HTemp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  1st layer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Computation of weighted features
H = permute(H,[4 1 2 3 5]);
H=reshape(H, [N size_pooled_map*size_pooled_map*AngelNum*k_size]);    
if N > size_pooled_map*size_pooled_map*AngelNum*k_size
    OutputWeight1=inv(eye(size(H',1))/c_rho+ H'  * H) * H' *  T';         
else
    OutputWeight1=H' * inv(eye(size(H,1))/c_rho+  H * H') *  T';         
end
TY=(H * OutputWeight1)'; 
weightedH = zeros(size(H));
MissClassificationRate_Training=0;
predLabels = zeros(N,1);
for i = 1 : N
    [x, label_index_expected]=max(T(:,i));
    [x, label_index_actual]=max(TY(:,i));
    weightedH(i,:) = H(i,:).*OutputWeight1(:,label_index_actual)';
    predLabels(i) = label_index_actual;
    if label_index_actual~=label_index_expected
         MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainingAccuracy1=1-MissClassificationRate_Training/N
H=weightedH;
H=reshape(H, [N size_pooled_map size_pooled_map AngelNum k_size]);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Computation of weighted features

% train SVM (one-against-all models)
model = cell(numLabels,1);
for k=1:numLabels
    model{k} = svmtrain(double(trainLabel==k),H, '-c 1 -g 0.2 -b 1');
end
clear H
TrainingTime=toc(train_time) 

%% Final step: Calculate the output of testing input
%load testing data
load modelNetTest128.mat
test_time=tic;
TV.P=imageTestData;
clear imageTestData
TV.T=testLabel';
TV.P=TV.P(:,:,:,1:TestNum);
TV.T=TV.T(:,1:TestNum);
clear X
clear Y
NumberofTestingData=size(TV.T,2);
InputDataLayerTest = TV.P;
clear TV.P 
N=NumberofTestingData;
numTest=N;
X= InputDataLayerTest(:,:,1:AngelNum,:,:);
clear InputDataLayerTest;
InputMask=maskTestData(:,:,:,1:TestNum);
clear maskTrainData;

%%%%%%%%%%%%%%%%%%%%%%%%%% 1st layer
C=zeros(size_map,size_map,AngelNum,N,k_size);
for k=1:k_size
    C(:,:,:,:,k)= convn(X, A(:,:,k),'same'); 
     C(:,:,:,:,k)= C(:,:,:,:,k)/sum(sum(A(:,:,k)));
    C(:,:,:,:,k) =C(:,:,:,:,k) .*InputMask;
end
clear A1Rep
clear X
InputMask_2=InputMask(1 : pool_size : end, 1 : pool_size : end, :,:);
clear InputMask
H=zeros(size_pooled_map,size_pooled_map, AngelNum,N,  k_size);
for k=1:k_size
     z=C(:,:,:,:,k);
     HTemp= z(1 : pool_size : end, 1 : pool_size : end, :,:);
     H(:,:,:,:,k)= HTemp;
end
clear C
clear z
clear HTemp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  1st layer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Computation of weighted features
H = permute(H,[4 1 2 3 5]);
H=reshape(H, [N size_pooled_map*size_pooled_map*AngelNum*k_size]);    
TY=(H * OutputWeight1)'; 
weightedH = zeros(size(H));
MissClassificationRate_Training=0;
predLabels = zeros(N,1);
for i = 1 : N
    label_index_expected=TV.T(:,i);
    [x, label_index_actual]=max(TY(:,i));
    weightedH(i,:) = H(i,:).*OutputWeight1(:,label_index_actual)';
    predLabels(i) = label_index_actual;
    if label_index_actual~=label_index_expected
         MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TestAccuracy1=1-MissClassificationRate_Training/N
H=weightedH;
H=reshape(H, [N size_pooled_map size_pooled_map AngelNum k_size]);   
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Computation of weighted features

% SVM testing (get probability estimates of test instances using each model)
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(testLabel==k), H, model{k}, '-b 1');
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end

% predict the class with the highest probability
[~,pred] = max(prob,[],2);
TestingTime=toc(test_time)
clear H
%clear OutputWeight
%%%%%%%%%% Calculate training & testing classification accuracy 
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
         label_index_expected=TV.T(:,i);
         label_index_actual=pred(i);
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    SVMTestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)
