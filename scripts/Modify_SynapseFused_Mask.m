close all;
clear all;
clc;


%% Create synapse and junk balanced mask for headless training
im_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_SynapseFused_Junk.tif';
info_im=imfinfo(im_file);
david_mask=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint8');
for slice=1:numel(info_im)
    david_mask(:,:,slice)=imread(im_file,slice);
end
[~,~,raw]=xlsread('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448_annotation_update.xlsx');

% Adopt labeled background here
mask_headless_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_Headless_Balanced.tif';
mask_headless=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint8');
for slice=1:numel(info_im)
    mask_headless(:,:,slice)=imread(mask_headless_file,slice);
end

num_synapse=0;
num_junk=0;
for i=2:size(raw,1)
    if strcmp(raw{i,2},'synapse')
        num_synapse=num_synapse+1;
    elseif strcmp(raw{i,2},'junk')
        num_junk=num_junk+1;
    end
end
step=round(num_synapse/num_junk);

bw_david_mask=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint8');
bw_david_mask(david_mask~=0)=1;
CC=bwconncomp(bw_david_mask);
n_synapse=0;
for obj=1:CC.NumObjects
    curr_idx=CC.PixelIdxList{obj};
    curr_obj=david_mask(curr_idx);
    curr_obj_val=max(curr_obj);
    if curr_obj_val==255
        n_synapse=n_synapse+1;
        if mod(n_synapse,step)~=0
            david_mask(curr_idx)=0;
        end
    end
end

final_mask=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint8');
final_mask(mask_headless==1)=1;
final_mask(david_mask==255)=3;
final_mask(david_mask==127)=2;


fn_out='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_Headless_Balanced_SynapseFused.tif';
for slice=1:numel(info_im)
    if slice==1
        imwrite(final_mask(:,:,slice),fn_out);
    else
        imwrite(final_mask(:,:,slice),fn_out,'WriteMode','append');
    end
end

imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t = Tiff(fn_out,'r+');
for count=1:numel(info_im)
   setDirectory(t,count)
   setTag(t,Tiff.TagID.ImageDescription, imageDescription);
   rewriteDirectory(t);
end
close(t)


%% Create a labeled mask with labeled synapse=255,junk=127,edge=63,others=0
im_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_SynapseFused_Junk.tif';
info_im=imfinfo(im_file);
david_mask=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint8');
for slice=1:numel(info_im)
    david_mask(:,:,slice)=imread(im_file,slice);
end
[~,~,raw]=xlsread('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448_annotation_update.xlsx');
mask_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448-lillvis-manual/';
for i=2:size(raw,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    
    if strcmp(raw{i,2},'edge')
        msk=zeros(info_im(1).Height,info_im(1).Width,numel(info_im),'uint16');
        for slice=1:numel(info_im)
            msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice);
        end
        david_mask(msk~=0)=63;
    end
    
end
fn_out='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Mask_SynapseFused_Junk_Edge.tif';
for slice=1:numel(info_im)
    if slice==1
        imwrite(david_mask(:,:,slice),fn_out);
    else
        imwrite(david_mask(:,:,slice),fn_out,'WriteMode','append');
    end
end

imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t = Tiff(fn_out,'r+');
for count=1:numel(info_im)
   setDirectory(t,count)
   setTag(t,Tiff.TagID.ImageDescription, imageDescription);
   rewriteDirectory(t);
end
close(t)

