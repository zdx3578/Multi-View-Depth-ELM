% store the image label
% 1 红色 2 绿色 3 蓝色 4 紫色 5 蓝色 6 
colorDict = [
	1,0,0; 0,1,0;0,0,1;1,0,1;0,1,1;1,1,0;...
	1,.5,.5;.5,1,.5;.5,.5,1;1,.5,1;.5,1,1;1,1,.5;...
	.5,0,0;0,.5,0;0,0,.5;.5,0,.5;0,.5,.5;.5,.5,0;...
    	1,.3,.7;1,.7,.3;.7,1,.3;.3,1,.7;.7,.3,1;.3,.7,1
];
colorDict=colorDict*255;
res=128;
categoryName='Airplane';
AngleNum=19;
saveFileName=[categoryName mat2str(AngleNum)];
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
                            %disp(k)
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
                    for k=1:6
                        if isequal(colorValue,colorDict(k,:))
                            TestImgLabel(ii,j,m,i)=k;
                            %disp(k)
                        end
                    end
                end
            end
        end
    end
end


save(saveFileName, 'dataOuputTrain1','dataOuputTest1','maskOutput1','maskTestOutput1','TrainImgLabel','TestImgLabel');