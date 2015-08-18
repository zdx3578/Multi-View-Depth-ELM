% Generate weightedFeatures128_20TestAll.mat;
 %%%%%%%%%%%%%%%%%%%%%%%%%% 
clear all
load modelNetTest128_20.mat; 
P=imageTestData;
clear imageTestData
T=testLabel';
clear testLabel
TrainNum=908;
TestNum=TrainNum
P=P(:,:,:,1:TrainNum)/255; 
T=T(:,1:TrainNum); 
NumberofTrainingData=size(P,4);  
number_class=3;
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
AngelNum=20; 
input_size=128;
inputDim=input_size;
r_size=4;
pool_size=1;
k_size=1;
size_map= input_size; 
size_pooled_map =ceil( (size_map)/pool_size);
sizeRep = size_map / r_size;
InputMask=maskTestData(:,:,:,1:TrainNum);
clear maskTestData;
InputMask=InputMask(:,:,1:AngelNum,:,:);
c_rho=0.01;
N=NumberofTrainingData;
sampleNum=N;
InputDataLayer = P;
clear P;
train_time=tic;
%%%%%%%%%%%%%%%%%%%%%%%%%% 1st layer
InputWeight = zeros(r_size* r_size, k_size);
for i=1:k_size
    randomWeight_r=rand(r_size,r_size); 
    randomWeight_r = reshape(randomWeight_r, [r_size*r_size 1]);
    InputWeight(:,i)=randomWeight_r ;      
end
A =reshape(InputWeight, [r_size r_size k_size]);
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
H=C;
clear C
clear z
clear HTemp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  1st layer
size_pooled_map_4=size_pooled_map;
H=reshape(H, [N size_pooled_map_4*size_pooled_map_4*AngelNum*k_size]);    
%%%%%%%%%%%%%%%%%%%%%%%%%% 
% output-weight
if N > size_pooled_map_4*size_pooled_map_4*AngelNum*k_size
    OutputWeight=inv(eye(size(H',1))/c_rho+ H'  * H) * H' *  T';         
else
    OutputWeight=H' * inv(eye(size(H,1))/c_rho+  H * H') *  T';         
end
TrainingTime=toc(train_time) 
TY=(H * OutputWeight)';
weightedFeatures = zeros(size(H));
MissClassificationRate_Training=0;
predLabels = zeros(sampleNum,1);
for i = 1 : sampleNum
    [x, label_index_expected]=max(T(:,i));
    [x, label_index_actual]=max(TY(:,i));
    weightedFeatures(i,:) = H(i,:).*OutputWeight(:,label_index_actual)';
    predLabels(i) = label_index_actual;
    if label_index_actual~=label_index_expected
         MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainingAccuracy=1-MissClassificationRate_Training/sampleNum
save weightedFeatures128_20TestAll.mat weightedFeatures -v7.3;
