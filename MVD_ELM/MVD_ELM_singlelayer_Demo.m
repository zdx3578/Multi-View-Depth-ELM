% MVD-ELM for 3D shape classification
% single layer ( 6 views) demo.
 %%%%%%%%%%%%%%%%%%%%%%%%%% 
clear all
load modelNetTrain128.mat;
TrainNum=3991;
TestNum=908;
% Load training dataset 
P=imageTrainData;
clear imageTrainData
T=trainLabel';
clear trainLabel
P=P(:,:,:,1:TrainNum); 
T=T(:,1:TrainNum); 
NumberofTrainingData=size(P,4);  
number_class=10;
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
pool_size=8;
k_size=2;
size_map= input_size; 
size_pooled_map =ceil( (size_map)/pool_size);
sizeRep = size_map / r_size;
InputMask=maskTrainData(:,:,:,1:TrainNum);
clear maskTrainData;
InputMask=InputMask(:,:,1:AngelNum,:,:);
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
%  Second step: Convulational node computation and summerization
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
size_pooled_map_4=size_pooled_map;
H=reshape(H, [size_pooled_map_4,size_pooled_map_4,AngelNum,N*k_size]);
H=reshape(H, [size_pooled_map_4*size_pooled_map_4*AngelNum,N*k_size]);
H=H';
H=reshape(H, [N size_pooled_map_4*size_pooled_map_4*AngelNum*k_size]);    
%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Fourth step: output-weight
if N > size_pooled_map_4*size_pooled_map_4*AngelNum*k_size
    OutputWeight=inv(eye(size(H',1))/c_rho+ H'  * H) * H' *  T';         
else
    OutputWeight=H' * inv(eye(size(H,1))/c_rho+  H * H') *  T';         
end
clear H
TrainingTime=toc(train_time) 

%% Final step: Calculate the output of testing input
% load testing data
load modelNetTest128.mat
test_time=tic;
TV.P=imageTestData;
clear imageTestData
TV.T=testLabel';
clear testLabel
TV.P=TV.P(:,:,:,1:TestNum);
TV.T=TV.T(:,1:TestNum);
clear X
clear Y
NumberofTestingData=size(TV.T,2);
InputDataLayerTest = TV.P;
clear TV.P 
N=NumberofTestingData;
X= InputDataLayerTest(:,:,1:AngelNum,:,:);
clear InputDataLayerTest;
InputMaskTest=maskTestData(:,:,:,1:TestNum);
clear maskTrainData;
InputMaskTest=InputMaskTest(:,:,1:AngelNum,:,:);
C=zeros(size_map,size_map,AngelNum,N,k_size);
for k=1:k_size
    C(:,:,:,:,k)= convn(X, A(:,:,k),'same'); 
    C(:,:,:,:,k)= C(:,:,:,:,k)/sum(sum(A(:,:,k)));
    C(:,:,:,:,k) =C(:,:,:,:,k) .*InputMaskTest;
end
clear A1Rep
clear X
clear InputMaskTest
H=zeros(size_pooled_map,size_pooled_map, AngelNum,N,  k_size);
for k=1:k_size
     z=C(:,:,:,:,k);
     HTemp= z(1 : pool_size : end, 1 : pool_size : end, :,:);
     H(:,:,:,:,k)= HTemp;
end
clear C
clear z
clear HTemp
size_pooled_map_4=size_pooled_map;
H=reshape(H, [size_pooled_map_4,size_pooled_map_4,AngelNum,N*k_size]);
H=reshape(H, [size_pooled_map_4*size_pooled_map_4*AngelNum,N*k_size]);
H=H';
H=reshape(H, [N size_pooled_map_4*size_pooled_map_4*AngelNum*k_size]);
% Computing label predictions
TY=(H*OutputWeight)'; 
TestingTime=toc(test_time)
clear H
%%%%%%%%%% Calculate training & testing classification accuracy 
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2);
    for i = 1 : size(TV.T, 2)
         label_index_expected=TV.T(:,i);
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)