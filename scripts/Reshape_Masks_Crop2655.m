close all;
clear all;
clc;

file_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/C1_2655-4788-5446_BGsubtract.tif';
im_info=imfinfo(file_2655);
rows=im_info(1).Height;
cols=im_info(1).Width;
slices=numel(im_info);

im_2655=zeros(rows,cols,slices,'uint16');
for i=1:slices
    im_2655(:,:,i)=imread(file_2655,i);
end

level=multithresh(im_2655,3);
quant_2655=imquantize(im_2655,level);
im_thresholded=zeros(rows,cols,slices,'uint8');
im_thresholded(quant_2655>1)=1;

mask_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/Mask_Synapse_Junk.tif';
mask_2655=zeros(rows,cols,slices,'uint8');
for i=1:slices
    mask_2655(:,:,i)=imread(mask_file,i);
end

mask_new=mask_2655.*im_thresholded;
mask_new_name='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/Mask_Synapse_Junk_narrowed.tif';
for i=1:slices
    if i==1
        imwrite(mask_new(:,:,i),mask_new_name);
    else
        imwrite(mask_new(:,:,i),mask_new_name,'WriteMode','append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t=Tiff(mask_new_name,'r+');
for slice=1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t)