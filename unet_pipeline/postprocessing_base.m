clear all;
close all;
clc;


data_path = '/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/For_check/';
img_name = 'prediction_unet_2.tif';

tic;
img = read_tif([data_path,img_name]);
img_close = closing_img(img);
img_close_watershed = WaterShed(img_close);
write_tif(img_close_watershed, [data_path, 'watershed_close_', img_name]);
toc;

% img_watershed = WaterShed(img);
% write_tif(img_watershed, [data_path, 'watershed_', img_name]);


function img = read_tif(img_file)
% Read tif image
im_info = imfinfo(img_file);
rows = im_info(1).Height;
cols = im_info(1).Width;
slices = numel(im_info);
img = zeros(rows, cols, slices, 'uint16');
for slice = 1:slices
    img(:,:,slice)=imread(img_file, slice);
end
end


function write_tif(data,name)
% Write data into tif image
slices = size(data,3);
for slice = 1:slices
    if slice == 1
        imwrite(data(:,:,slice),name);
    else
        imwrite(data(:,:,slice),name,'WriteMode','append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',slices,slices);
t = Tiff(name,'r+');
for slice = 1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);
end


function seg_img = closing_img(img)
fprintf('Image closing\n')
img(img~=0)=1;
SE=strel('sphere',3);
seg_img=imclose(img,SE);
% % image filling
% seg_img=imfill(seg_img,'holes');
seg_img(seg_img~=0) = 255;
end


function seg_img=WaterShed(bw)
% Watershed segmentation
fprintf('Watershed segmentation\n');
bw(bw~=0)=1;
D=-bwdist(~bw);
mask=imextendedmin(D,2);
D2=imimposemin(D,mask);
Ld=watershed(D2);
seg_img=bw;
seg_img(Ld==0)=0;
% % Remove blobs that are smaller than 100 voxels
% CC=bwconncomp(seg_img);
% for obj=1:CC.NumObjects
%     obj_idx=CC.PixelIdxList{obj};
%     if length(obj_idx)<100
%         seg_img(obj_idx)=0;
%     end
% end
seg_img(seg_img~=0)=255;
end