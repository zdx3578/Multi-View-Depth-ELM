% prepare data for mesh segmentation: 1. depth 2. label. 3 mask.
% test mesh number
exNum=testIndex;
saveName= [ className '_ex_' mat2str(exNum) '.mat']; 
TestFiles = allMeshNum(testIndex) 
TrainFiles =setdiff(allMeshNum,TestFiles) 

[dataOuputTrain1, maskOutput1] = readImageAndLabelForSeg(TrainFiles, Angle,resolution); 
[colorOuputTrain1, maskOutput1_test] = readImageAndLabelForColor(TrainFiles, Angle,resolution); 

[dataOuputTest1, maskTestOutput1] = readImageAndLabelForSeg(TestFiles, Angle,resolution); 
[colorOuputTest1, maskTestOutput1_test] = readImageAndLabelForColor(TestFiles, Angle,resolution); 

colorDict = [
	1,0,0; 0,1,0;0,0,1;1,0,1;0,1,1;1,1,0;...
	1,.5,.5;.5,1,.5;.5,.5,1;1,.5,1;.5,1,1;1,1,.5;...
	.5,0,0;0,.5,0;0,0,.5;.5,0,.5;0,.5,.5;.5,.5,0;...
    1,.2,.2;.2,1,.2;.2,.2,1;1,.2,1;.2,1,1;1,1,.2;...
	.2,0,0;0,.2,0;0,0,.2;.2,0,.2;0,.2,.2;.2,.2,0;...
    1,.4,.4;.4,1,.4;.4,.4,1;1,.4,1;.4,1,1;1,1,.4;...
	.4,0,0;0,.4,0;0,0,.4;.4,0,.4;0,.4,.4;.4,.4,0;...
    1,.6,.6;.6,1,.6;.6,.6,1;1,.6,1;.6,1,1;1,1,.6;...
	.6,0,0;0,.6,0;0,0,.6;.6,0,.6;0,.6,.6;.6,.6,0;
];
colorDict=colorDict*255;
res=resolution;
AngleNum=Angle;
LengthFiles = length(TrainFiles);
TrainImgLabel=zeros(res,res,AngleNum,LengthFiles);
for m=1:AngleNum
    for i = 1:LengthFiles;
    colorI=colorOuputTrain1(:,:,:,m,i);
        for ii=1:res
            for j=1:res
                colorValue=colorI(ii,j,:);
                colorValue=reshape(colorValue,[1 3]);
                if  ~isequal(colorValue,[255 255 255])
                    for k=1:6
                        if isequal(colorValue,colorDict(k,:))
                            TrainImgLabel(ii,j,m,i)=k;
                        end
                    end
                end
            end
        end
    end
end

LengthFiles = length(TestFiles);
TestImgLabel=zeros(res,res,AngleNum,LengthFiles);
for m=1:AngleNum
    for i = 1:LengthFiles;
    colorI=colorOuputTest1(:,:,:,m,i);
        for ii=1:res
            for j=1:res
                colorValue=colorI(ii,j,:);
                colorValue=reshape(colorValue,[1 3]);
                if  ~isequal(colorValue,[255 255 255])
                    for k=1:50
                        if isequal(colorValue,colorDict(k,:))
                            TestImgLabel(ii,j,m,i)=k;
                        end
                    end
                end
            end
        end
    end
end

save(saveName, 'dataOuputTrain1','dataOuputTest1','maskOutput1','maskTestOutput1','TrainImgLabel','TestImgLabel');