% MVD-ELM for 3D shape classification
% two layers (6 views) demo.
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
r_size=6;
r_size_2=4;
pool_size=4;
k_size=2;
k_size_2=2;
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
InputMask_2=InputMask(1 : pool_size : end, 1 : pool_size : end, :,:);
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

%%%%%%%%%%%%%%%%%%%%%%%%%% 2nd layer
X_2= H;
%  First step: generate random  convulational kernels
for j=1:k_size_2
    for i=1:k_size
        randomWeight_r=rand(r_size_2,r_size_2); 
        A2Temp{i}=randomWeight_r;  
    end
    A2{j}=A2Temp;
end
%  Second step: Convulational computation 
C=zeros(size_map_2,size_map_2,AngelNum, N,k_size,k_size_2);
for j=1:k_size_2
    A2Temp = A2{j};
    for k=1:k_size
        C(:,:,:,:,k,j)= convn(X_2(:,:,:,:,k), A2Temp{k},'same'); 
        C(:,:,:,:,k,j)= C(:,:,:,:,k,j)/sum(sum(A2Temp{k}));
        C(:,:,:,:,k,j)=C(:,:,:,:,k,j).*InputMask_2;
    end
end
clear A1Rep
clear X
clear InputMask
% Third step:  pooling 
H=zeros(size_pooled_map_2,size_pooled_map_2, AngelNum,N, k_size,k_size_2);
for j=1:k_size_2
    for k=1:k_size
         z=C(:,:,:,:,k,j);
         HTemp= z(1 : pool_size_2 : end, 1 : pool_size_2 : end, :,:);
         H(:,:,:,:,k,j)= HTemp;
    end
end
clear C
clear z
clear HTemp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2nd layer
H=reshape(H, [size_pooled_map_2,size_pooled_map_2,AngelNum,N*k_size*k_size_2]);
H=reshape(H, [size_pooled_map_2*size_pooled_map_2*AngelNum,N*k_size*k_size_2]);
H=H';
H=reshape(H, [N size_pooled_map_2*size_pooled_map_2*AngelNum*k_size*k_size_2]);    
%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Fourth step: output-weight
if N > size_pooled_map_2*size_pooled_map_2*AngelNum*k_size
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

%%%%%%%%%%%%%%%%%%%%%%%%%% 2nd layer
X_2= H;
    for j=1:k_size_2
        for i=1:k_size
            randomWeight_r=rand(r_size_2,r_size_2); 
            A2Temp{i}=randomWeight_r;  
        end
        A2{j}=A2Temp;
    end
    C=zeros(size_map_2,size_map_2,AngelNum, N,k_size,k_size_2);
    for j=1:k_size_2
        A2Temp = A2{j};
        for k=1:k_size
            C(:,:,:,:,k,j)= convn(X_2(:,:,:,:,k), A2Temp{k},'same'); 
            C(:,:,:,:,k,j)= C(:,:,:,:,k,j)/sum(sum(A2Temp{k}));
            C(:,:,:,:,k,j)=C(:,:,:,:,k,j).*InputMask_2;
        end
    end
    clear A1Rep
    clear X
    clear InputMask
    H=zeros(size_pooled_map_2,size_pooled_map_2, AngelNum,N, k_size,k_size_2);
    for j=1:k_size_2
        for k=1:k_size
             z=C(:,:,:,:,k,j);
             HTemp= z(1 : pool_size_2 : end, 1 : pool_size_2 : end, :,:);
             H(:,:,:,:,k,j)= HTemp;
        end
    end
    clear C
    clear z
    clear HTemp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2nd layer
H=reshape(H, [size_pooled_map_2,size_pooled_map_2,AngelNum,N*k_size*k_size_2]);
H=reshape(H, [size_pooled_map_2*size_pooled_map_2*AngelNum,N*k_size*k_size_2]);
H=H';
H=reshape(H, [N size_pooled_map_2*size_pooled_map_2*AngelNum*k_size*k_size_2]);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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