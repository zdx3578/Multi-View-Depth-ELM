clear
% load weightedFeatures
load weightedFeatures128_20TestAll.mat;
AllFiles = dir(fullfile('F:\3DDL_PG2015_Code\MVD_ELM\Visualization\bedSubData','*.off'));
meshStart=0;
AngleNum=20;
resolution=128;
meshTotalStart=50;  % the label start index number of bed 
k_size=1;
angleNum=19;
W=reshape(weightedFeatures,[908 128 128 AngleNum k_size]);
for i=1:100
   % load mesh
   filename=AllFiles(i).name;
   [pathstr,name,ext] = fileparts(filename); 
    loadname=['imageLableAll' name];
    load(loadname)
    [M, fcount] = mesh_read_off(filename );  
    newM= mesh_normalize(M);
    [VIF] = mesh_get_con(M);
    F1=newM.faces; 
    num_faces = size(F1,1)
    num_verts= size(newM.vertices,1);
    X1=newM.vertices+1;
	% color matrix
    ColorMatrix=zeros(num_faces,1);
    ColorMVert=zeros(num_verts,1);
    meshIndex=meshStart+i;
    for j=1:angleNum
        if j>10
        imageName=[name '_findex_ 0_' num2str(j-1)  '.bmp'];
        else
        imageName=[name '_findex_ 0_0' num2str(j-1)  '.bmp'];
        end
        featureMap=imread(imageName);
        figure(1*angleNum)
        imshow(featureMap,[])
        
        figure(2*angleNum)
        featureMap=W(meshTotalStart+i,:,:,j,1);
        featureMap=squeeze(featureMap);
        mID=(featureMap>0);
        featureMap=featureMap.*mID;
        imshow(featureMap,[]);
        imagelabel=imagelabelAll(:,:,j);
		% compute activation for every triangle
        for xi=1:128
            for yi=1:128
                if imagelabel(xi,yi) ~=0 && featureMap(xi,yi)~=0
                ColorMatrix(imagelabel(xi,yi)) = ColorMatrix(imagelabel(xi,yi))+ featureMap(xi,yi);
                end
            end
        end
    end
    % compute activation for every vertex 
  for vindex=1:num_verts
        vFaceIndexs=VIF(vindex);
        vFaceIndexs=vFaceIndexs{1,1};
        if size(vFaceIndexs,1)>0
            for vFaceindex=1:size(vFaceIndexs,1)
                    ColorMVert(vindex)=ColorMVert(vindex)+ColorMatrix(vFaceIndexs(vFaceindex));
            end
        end
  end   
    % show activation
    figure(i*10);
    colormap(jet)
    set(gcf,'Renderer','OpenGL')
    set(trisurf(F1, X1(:,1), X1(:,2), X1(:,3),ColorMVert),'FaceLighting','phong','FaceColor','interp',  'LineStyle', 'none',  'AmbientStrength',0.5);
    colorbar
    light('Position',[1 0 0],'Style','infinite');
    light('Position',[0 0 1],'Style','infinite');
    light('Position',[0 1 0],'Style','infinite');
    axis equal
    axis off
    img_file = [num2str(meshTotalStart+i) '.jpg'];
    f = getframe(gcf); %# Capture the current window 
    imwrite(f.cdata,img_file); %# Save the frame data
end

