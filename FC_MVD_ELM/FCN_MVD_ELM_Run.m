% FCN-MVD-ELM for 3D shape segmentation. 
% Segment testing depth image
 %%%%%%%%%%%%%%%%%%%%%%%%%% load data
close all
load Ant_ex_1.mat;
totalAngleNum=2
for AngleNum=1:totalAngleNum
    Resolution=128;
    TrainNum=16;
    P=dataOuputTrain1(:,:,AngleNum,:);
    P=squeeze(P);
    Q=maskOutput1(:,:,AngleNum,:);
    Q=squeeze(Q);
    T=TrainImgLabel(:,:,AngleNum,:);
    T=squeeze(T);
    NumberofTrainingData=size(P,3);  
    number_class=6;

    % Set the size stuff 
    input_size=Resolution;
    d=input_size;
    r_size=4;
    pool_size=1;
    pool_size_2=1;
    k_size=2;
    k_size_2=2;
    size_map= input_size; 
    size_pooled_map =ceil( (size_map)/pool_size);
    sizeRep = size_map / r_size;
    c_rho=0.01;
    N=NumberofTrainingData;

    InputDataLayer = P;
    clear P;
    train_time=tic;
    %%%%%%%%%%%%%%%%%%%%%%%%%% 1st layer
       %  First step: generate random  convulational kernels
        for i=1:k_size
            randomWeight_r= randsrc(r_size,r_size,[1 2  3 4 -1 -2 -3 -4]); 
            A1{i}=randomWeight_r;  
        end

        %  Second step: Convulational computation
        X= InputDataLayer;
        X= reshape(X, [Resolution Resolution N]);
        clear InputDataLayer;

        Q=reshape(Q,[Resolution Resolution N]);
        InputMask=Q;
        clear Q;
        C=zeros(size_map,size_map,N,k_size);
        for k=1:k_size
            A1Rep=repmat(A1{k}, [sizeRep sizeRep N]);
            C(:,:,:,k)=X + A1Rep;
            C(:,:,:,k)=C(:,:,:,k).*InputMask;
        end
        clear A1Rep
        clear X
        % Third step:  pooling
        H=zeros(size_pooled_map,size_pooled_map, N,  k_size);
        T1=T(1 : pool_size : end, 1 : pool_size : end, :); 
        for k=1:k_size
             z=C(:,:,:,k);
             HTemp= z(1 : pool_size : end, 1 : pool_size : end, :,:);
             H(:,:,:,k)= HTemp;
        end
        clear C
        clear z
        clear HTemp
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   1st layer

    %%%%%%%%%%%%%%%%%%%%%%%%%% 2nd layer
        %  First step: generate random  convulational kernels
        size_map= size_pooled_map; 
        size_pooled_map =ceil( (size_map)/pool_size_2);
        sizeRep = size_map / r_size;
        for j=1:k_size_2
            for i=1:k_size
                randomWeight_r= randsrc(r_size,r_size,[1 2  3 4 -1 -2 -3 -4]); 
                A2Temp{i}=randomWeight_r;  
            end
            A2{j}=A2Temp;
        end

        %  Second step: Convulational computation 
        InputMask=InputMask(1 : pool_size : end, 1 : pool_size : end, :,:);
        C=zeros(size_map,size_map,N,k_size,k_size_2);
        for j=1:k_size_2
            A2Temp = A2{j};
            for k=1:k_size
                A1Rep=repmat(A2Temp{k}, [sizeRep sizeRep  N]);
                C(:,:,:,k,j)=H(:,:,:,k) + A1Rep;
                C(:,:,:,k,j)=C(:,:,:,k,j).*InputMask;
            end
        end
        clear A1Rep
        clear X
        clear InputMask
        % Third step:  pooling
        H=zeros(size_pooled_map,size_pooled_map, N,  k_size,k_size_2);
        T2=T1(1 : pool_size_2 : end, 1 : pool_size_2 : end,:); 
        for j=1:k_size_2
            for k=1:k_size
                 z=C(:,:,:,k,j);
                 HTemp= z(1 : pool_size_2: end, 1 : pool_size_2 : end, :,:);
                 H(:,:,:,k,j)= HTemp;
            end
        end
        H=reshape(H,  [size_pooled_map size_pooled_map N  k_size*k_size_2] );
        clear C
        clear z
        clear HTemp
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   2nd layer

    % Generating masks
    LabelAllMap=zeros(size_pooled_map, size_pooled_map,N,number_class);
    MaskAllMap=zeros(size_pooled_map, size_pooled_map,N,number_class);
    for j=1:TrainNum
        TT=T2(:,:,j);
        for k=1:number_class
            TTemp=zeros(size_pooled_map, size_pooled_map);
            MaskTemp=zeros(size_pooled_map, size_pooled_map);
            idx=find(TT==k);
            TTemp(idx)=k;
            MaskTemp(idx)=1;
            TTemp=reshape(TTemp,[size_pooled_map size_pooled_map]);
            MaskTemp=reshape(MaskTemp,[size_pooled_map size_pooled_map]);
            LabelAllMap(:,:,j,k)=TTemp;
            MaskAllMap(:,:,j,k)=MaskTemp;
        end
    end
    % Masking feature maps
    HNewTotal=zeros(size_pooled_map, size_pooled_map,k_size,k_size_2,N,number_class);
    for m=1:k_size
        for n=1:k_size_2
            for j=1:NumberofTrainingData
                HTemp=H(:,:,j);
                for k=1:number_class
                    MaskTemp=MaskAllMap(:,:,j,k);
                    HNewTotal(:,:,m,n,j,k)=HTemp.*MaskTemp;
                end
            end
        end
    end

    H=HNewTotal;
    clear HNewTotal
    H=reshape(H, [size_pooled_map*size_pooled_map*k_size*k_size_2,N*number_class]);
    H=H';  

    T=LabelAllMap;
    T=reshape(T, [size_pooled_map*size_pooled_map N*number_class]);
    if N*number_class > k_size * k_size_2* size_map 
       OutputWeight=inv(eye(size(H',1))/c_rho+ H'  * H) * H' *  T';       
    else
       OutputWeight=H' * inv(eye(size(H,1))/c_rho+  H * H') *  T';          
    end
    clear T
    clear H
    TrainingTime=toc(train_time)  

    %% Final step: Calculate the output of testing input 
    test_time=tic;
    TV.P=dataOuputTest1(:,:,AngleNum,:);
    TV.P=squeeze(TV.P);
    TV.Q=maskTestOutput1(:,:,AngleNum,:);
    TV.Q=squeeze(TV.Q);
    TV.T=TestImgLabel(:,:,AngleNum,:);
    TV.T=squeeze(TV.T);
    NumberofTestingData=size(TV.P,3);
    InputDataLayerTest = TV.P;
    clear TV.P 
    N=NumberofTestingData;
    %%%%%%%%%%%%%%%%%%%%%%%%%%

    X= InputDataLayerTest;
    X= reshape(X, [Resolution Resolution N]);
    clear InputDataLayerTest;

    input_size=Resolution;
    d=input_size;
    size_map= input_size; 
    size_pooled_map =ceil( (size_map)/pool_size);
    sizeRep = size_map / r_size;
    c_rho=0.01;
    %%%%%%%%%%%%%%%%%%%%%%%%%% 1st layer
        TV.Q=reshape(TV.Q,[Resolution Resolution N]);
        InputMask= TV.Q; %(:,:,1:TrainNum);
        clear  TV.Q;
        C=zeros(size_map,size_map,N,k_size);
        for k=1:k_size
            A1Rep=repmat(A1{k}, [sizeRep sizeRep N]);
            C(:,:,:,k)=X + A1Rep;
            C(:,:,:,k)=C(:,:,:,k).*InputMask;
        end
        clear A1Rep
        clear X

        H=zeros(size_pooled_map,size_pooled_map, N,  k_size);
        for k=1:k_size
             z=C(:,:,:,k);
             HTemp= z(1 : pool_size : end, 1 : pool_size : end, :,:);
             H(:,:,:,k)= HTemp;
        end
        clear C
        clear z
        clear HTemp
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   1st layer

    %%%%%%%%%%%%%%%%%%%%%%%%%% 2nd layer
        size_map= size_pooled_map; 
        size_pooled_map =ceil( (size_map)/pool_size_2);
        sizeRep = size_map / r_size;

        InputMask=InputMask(1 : pool_size : end, 1 : pool_size : end, :,:);
        C=zeros(size_map,size_map,N,k_size,k_size_2);
        for j=1:k_size_2
            A2Temp = A2{j};
            for k=1:k_size
                A1Rep=repmat(A2Temp{k}, [sizeRep sizeRep  N]);
                C(:,:,:,k,j)=H(:,:,:,k) + A1Rep;
                C(:,:,:,k,j)=C(:,:,:,k,j).*InputMask;
            end
        end
        clear A1Rep
        clear X
        clear InputMask

        HTotal=zeros(size_pooled_map,size_pooled_map, N,  k_size,k_size_2);
        for j=1:k_size_2
            for k=1:k_size
                 z=C(:,:,:,k,j);
                 HTemp= z(1 : pool_size_2: end, 1 : pool_size_2 : end, :,:);
                 HToal(:,:,:,k,j)= HTemp;
            end
        end
        clear C
        clear z
        clear HTemp
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  2nd layer

    % Masking testing featue maps
    HNewTotalTest=zeros(size_pooled_map, size_pooled_map,k_size,k_size_2,N,number_class);
    TY=zeros(size_pooled_map*size_pooled_map,number_class);
    for trainIndex=1:NumberofTrainingData
        for m=1:k_size
            for n=1:k_size_2
                for j=1:NumberofTestingData
                    HTemp=HToal(:,:,j,m,n);
                    for k=1:number_class
                        MaskTemp=MaskAllMap(:,:,trainIndex,k);
                        HNewTotalTest(:,:,m,n,j,k)=HTemp.*MaskTemp;
                    end
                end
            end
        end

        H=HNewTotalTest;
        H=reshape(H, [size_pooled_map*size_pooled_map*k_size*k_size_2,N*number_class]);
        H=reshape(H, [size_pooled_map*size_pooled_map*k_size*k_size_2,N*number_class]);
        H=H';

        TY=TY+(H*OutputWeight)';
    end


    TestingTime=toc(test_time)
    TestLabel=TY;
    TrueLabel=TestImgLabel(:,:,AngleNum);
    TrueLabel= TrueLabel(1 : pool_size : end, 1 : pool_size : end);
    TrueLabel= TrueLabel(1 : pool_size_2 : end, 1 : pool_size_2 : end);

    figure(AngleNum)
    resRes=128/(pool_size*pool_size_2)
    imgOut=convertLabelToRgb(TrueLabel);
    imshow(imgOut)

    maskTest=maskTestOutput1(:,:,AngleNum);
    maskTest= maskTest((1 : pool_size : end), (1 : pool_size : end));
    maskTest= maskTest((1 : pool_size_2 : end), (1 : pool_size_2 : end));
    maskTest=reshape(maskTest, [1 size_pooled_map*size_pooled_map]);
    TrueLabel=reshape(TrueLabel, [1 size_pooled_map*size_pooled_map]);
    OKLabel=zeros(size_pooled_map*size_pooled_map,number_class );
    for index=1:size_pooled_map*size_pooled_map
        [value,labelNum]=max(TestLabel(index,:));
        if value~=0
        OKLabel(index,labelNum)=labelNum;
        end
    end
    TestLabel=sum(OKLabel,2);
    TestLabel=TestLabel.*maskTest';


    figure(AngleNum+totalAngleNum)
    labelImage=reshape(TestLabel,[resRes resRes]);
    imgOut=convertLabelToRgb(labelImage);
    imshow(imgOut)


    MissClassificationRate_Testing=0;
        for i = 1 : size_pooled_map*size_pooled_map
             label_index_expected=TrueLabel(i);
             label_index_actual=TestLabel(i);
            if label_index_actual~=label_index_expected
                MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            end
        end
    TestingAccuracy=double(1-MissClassificationRate_Testing/(size_pooled_map*size_pooled_map))

end



