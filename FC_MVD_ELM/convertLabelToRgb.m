%convert label to rgb image
function imgOut=convertLabelToRgb(imgIn)
res=size(imgIn,1);
inputImg=imgIn;
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

newImg=zeros(res,res,3);

        for ii=1:res
            for jj=1:res
                if inputImg(ii,jj)==0
                    newImg(ii,jj,:)=[255 255 255];
                else 
                    colorValue=colorDict(inputImg(ii,jj),:);
                    newImg(ii,jj,:)=colorValue;
                end
            end
        end

        imgOut=newImg;
end       

