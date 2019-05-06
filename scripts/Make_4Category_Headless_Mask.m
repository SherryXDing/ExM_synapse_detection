close all;
clear all;
clc;

mask_2655_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/Mask_Headless_Balanced.tif';
mask_4179_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_Headless_Balanced_SynapseFused.tif';

mask_info=imfinfo(mask_2655_file);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);

mask_2655=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    mask_2655(:,:,slice)=imread(mask_2655_file,slice);
end

mask_4179=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    mask_4179(:,:,slice)=imread(mask_4179_file,slice);
end

mask_4179(mask_4179==3)=5;
mask_4179(mask_4179==2)=4;

mask_left=[mask_2655(:,1:cols/2,:),mask_4179(:,1:cols/2,:)];
mask_right=[mask_2655(:,cols/2+1:end,:),mask_4179(:,cols/2+1:end,:)];

out_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_combine_2655-4179/';
for slice=1:slices
    if slice==1
        imwrite(mask_left(:,:,slice),[out_path,'maskBalanced_4category_left.tif']);
        imwrite(mask_right(:,:,slice),[out_path,'maskBalanced_4category_right.tif']);
    else
        imwrite(mask_left(:,:,slice),[out_path,'maskBalanced_4category_left.tif'],'WriteMode','append');
        imwrite(mask_right(:,:,slice),[out_path,'maskBalanced_4category_right.tif'],'WriteMode','append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t=Tiff([out_path,'maskBalanced_4category_left.tif'],'r+');
for slice=1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t=Tiff([out_path,'maskBalanced_4category_right.tif'],'r+');
for slice=1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);