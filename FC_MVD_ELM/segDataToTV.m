% sliding window for segmenation data
% save as P,T, TV.P, TV.T format;
% Author: zhigexie@gmail.com 
% For 3DDL SIG project.
clear
name='Airplane';  % Change the category name here!
load([name '.mat']);% Change the category name here!
sizeWindow=32;
All255=repmat(255,sizeWindow*sizeWindow,1);
centerIndex=sizeWindow*sizeWindow/2;
AngleIndex=3;   %在此改变角度
Angle=sprintf('%d',AngleIndex);
sizeWindowNum=sprintf('_%d',sizeWindow);

%%%%%%% Process Training data %%%%%%%%%%%%%
TrainImg=dataOuputTrain1;
clear dataOuputTrain1
TrainLabel=TrainImgLabel;
clear TrainImgLabel
TrainNum=size(TrainImg,4);
m=AngleIndex;
index=1;
dictW={};
maskW={};
labelW={};
for k=1:TrainNum   
    img= TrainImg(:,:,m,k); %      imread('61_seg_ 0_02.bmp');
    label=TrainLabel(:,:,m,k);
    mask=maskOutput1(:,:,m,k);
    slidingWs=im2col(img, [sizeWindow sizeWindow], 'sliding');
    slidingMasks=im2col(mask, [sizeWindow sizeWindow], 'sliding');
    slidingLabels=im2col(label, [sizeWindow sizeWindow], 'sliding');
    slidingNum=size(slidingWs,2);
    for i=1:slidingNum
        slidingW=slidingWs(:,i);
        slidingMask=slidingMasks(:,i);
        templabelW=slidingLabels(:,i);
        if ~isequal(slidingW,All255) && (slidingW(centerIndex) ~=255)
            dictW{index}=slidingW;
            maskW{index}=slidingMask;
            labelW{index}=templabelW(centerIndex);
            index=index+1;
        end
    end
end


sizeNum=size(dictW,2);
P=zeros(sizeWindow,sizeWindow,sizeNum);
Q=zeros(sizeWindow,sizeWindow,sizeNum);
T=zeros(1,sizeNum);
for i=1:sizeNum
    slidingW=dictW{i};
    slidingW=reshape(slidingW, [sizeWindow sizeWindow]);
    P(:,:,i)= slidingW;
    maskWw=maskW{i};
    maskWw=reshape(maskWw, [sizeWindow sizeWindow]);
    Q(:,:,i)= maskWw;
    T(i)= labelW{i};
end

%%%%%%% Process Tesing data %%%%%%%%%%%%%
TrainImg=dataOuputTest1;
clear dataOuputTest1
TrainLabel=TestImgLabel;
clear TestImgLabel
TestNum=size(TrainImg,4);
testindex=1;
dictW={};
maskW={};
labelW={};
for k=1:TestNum 
    img= TrainImg(:,:,m,k); %      imread('61_seg_ 0_02.bmp');
    label=TrainLabel(:,:,m,k);
    mask=maskOutput1(:,:,m,k);
    slidingWs=im2col(img, [sizeWindow sizeWindow], 'sliding');
    slidingMasks=im2col(mask, [sizeWindow sizeWindow], 'sliding');
    slidingLabels=im2col(label, [sizeWindow sizeWindow], 'sliding');
    slidingNum=size(slidingWs,2);
    for i=1:slidingNum
        slidingW=slidingWs(:,i);
        slidingMask=slidingMasks(:,i);
        templabelW=slidingLabels(:,i);
        if ~isequal(slidingW,All255) && (slidingW(centerIndex) ~=255)
            dictW{testindex}=slidingW;
            maskW{testindex}=slidingMask;
            labelW{testindex}=templabelW(centerIndex);
            testindex=testindex+1;
        end
    end
end

sizeNum=size(dictW,2);
TV.P=zeros(sizeWindow,sizeWindow,sizeNum);
TV.Q=zeros(sizeWindow,sizeWindow,sizeNum);
TV.T=zeros(1,sizeNum);
for i=1:sizeNum
    slidingW=dictW{i};
    slidingW=reshape(slidingW, [sizeWindow sizeWindow]);
    TV.P(:,:,i)= slidingW;
    maskWw=maskW{i};
    maskWw=reshape(maskWw, [sizeWindow sizeWindow]);
    TV.Q(:,:,i)= maskWw;
    TV.T(i)= labelW{i};
end

saveName=[name 'TrainTestData' Angle sizeWindowNum '.mat'];
save(saveName,'T','P','Q','TV');
clear