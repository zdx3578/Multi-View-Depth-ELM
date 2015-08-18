function [imageDataOutput, imageMaskOutput] = readImageAndLabelForColor(FilesInput,AngleInput,resolutionInput)
    Files = FilesInput;
    AngleNum=AngleInput;
    resolution=resolutionInput;
    LengthFiles = length(Files);
    ImgNum=LengthFiles;
        Img=zeros(resolution,resolution,3,AngleNum,ImgNum);
       ImgM=zeros(resolution,resolution,AngleNum,ImgNum);
    for m=1:AngleNum
           for i = 1:LengthFiles;
              %filename=Files(i).name;
             %[pathstr,name,ext] = fileparts(filename); 
             name=Files(i);
             if m>9
                imgFileName=sprintf('%dnew_ 0_%d_color.bmp', name,m);  %m-1
             else
                imgFileName=sprintf('%dnew_ 0_0%d_color.bmp', name,m); %m-1
             end
             tempImg = imread(imgFileName);
             tempImg = imresize(tempImg, [resolution resolution]);
             tempMask = ImgTrans255Mask( double(tempImg(:,:,1)),resolution);

            Img(:,:,:,m,i)=tempImg;
            ImgM(:,:,m,i)=tempMask;
           end
    end
    imageDataOutput =  Img;
    imageMaskOutput = ImgM;
end