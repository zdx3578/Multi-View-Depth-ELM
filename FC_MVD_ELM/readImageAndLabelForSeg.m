function [imageDataOutput, imageMaskOutput] = readImageAndLabelForSeg(FilesInput,AngleInput,resolutionInput)
    Files = FilesInput;
    AngleNum=AngleInput;
    resolution=resolutionInput;
    LengthFiles = length(Files);
    ImgNum=LengthFiles;
        Img=zeros(resolution,resolution,AngleNum,ImgNum);
       ImgM=zeros(resolution,resolution,AngleNum,ImgNum);  
    for m=1:AngleNum
           for i = 1:LengthFiles;
             %filename=Files(i).name;
             %[pathstr,name,ext] = fileparts(filename); 
             name=Files(i);
             if m>10
                imgFileName=sprintf('%dnew_ 0_%d.bmp', name,m-1);
             else
                imgFileName=sprintf('%dnew_ 0_0%d.bmp', name,m-1);
             end
             tempImg = imread(imgFileName);
             tempImg = imresize(tempImg, [resolution resolution]);
                if(size(tempImg,3)==3)
                   tempImg = double(tempImg(:,:,1));
                    tempMask = ImgTrans255Mask(tempImg,resolution);
                    %thresh=1;
                    %tempMask=im2bw(tempImg,thresh);
                end
            Img(:,:,m,i)=tempImg;
            ImgM(:,:,m,i)=tempMask;
           end
    end
    imageDataOutput =  Img;
    imageMaskOutput = ImgM;
end