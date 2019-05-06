close all;
clear all;
clc;

% Postprocessing on pixcel calssification results
% Blob size < 250 --> junk
% Otherwise, do image closing and filling on synapses and junks separately, then voting based on number of synapse/junk pixels in the blob
% After that, based on voted image, get junks as final junks, use previously image closed and filled synapses as final synapses

data_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/';
% seg_file={[data_path,'GroundTruth_5631-2932-1842/C2_5631-2932-1842_pxl_cls_trainby2655-4179.tiff'],...
%     [data_path,'GroundTruth_5753-2694-2876/C2_5753-2694-2876_pxl_cls_trainby2655-4179.tiff'],...
%     [data_path,'GroundTruth_6127-2377-11480/C2_6127-2377-11480_pxl_cls_trainby2655-4179.tiff']};
% 
% final_name={[data_path,'GroundTruth_5631-2932-1842/C2_5631-2932-1842_pxlcls_postproc.tif'],...
%     [data_path,'GroundTruth_5753-2694-2876/C2_5753-2694-2876_pxlcls_postproc.tif'],...
%     [data_path,'GroundTruth_6127-2377-11480/C2_6127-2377-11480_pxlcls_postproc.tif']};
% 
% synapse_name={[data_path,'GroundTruth_5631-2932-1842/C2_5631-2932-1842_pxlcls_postproc_syn.tif'],...
%     [data_path,'GroundTruth_5753-2694-2876/C2_5753-2694-2876_pxlcls_postproc_syn.tif'],...
%     [data_path,'GroundTruth_6127-2377-11480/C2_6127-2377-11480_pxlcls_postproc_syn.tif']};
% 
% for i=1:numel(seg_file)
%     [voted_img,final_img]=PostprocessPxlClass(seg_file{i});
%     ImageWrite(final_img,final_name{i});
%     synapse_img=final_img;
%     synapse_img(synapse_img==127)=0;
%     synapse_final=WaterShed(synapse_img);
%     ImageWrite(synapse_final,synapse_name{i});
% end


seg_file=[data_path,'Output_combine_2655-4179//Feature_Test/Pxl_SelectedFeatures1_TrainLeft_TestRight.tiff'];
[voted_img,final_img]=PostprocessPxlClass(seg_file);
ImageWrite(voted_img,[data_path,'Output_combine_2655-4179/Pxl_SelectedFeature_TrainLeft_TestRight_postproc.tif']);
synapse_img=voted_img;
synapse_img(synapse_img==127)=0;
synapse_final=WaterShed(synapse_img);
ImageWrite(synapse_final,[data_path,'Output_combine_2655-4179/Pxl_SelectedFeature_TrainLeft_TestRight_postproc_syn.tif']);


%% Functions
function [voted_img,final_img]=PostprocessPxlClass(seg_file)

im_info=imfinfo(seg_file);
rows=im_info(1).Height;
cols=im_info(1).Width;
slices=numel(im_info);

seg_img=zeros(rows,cols,slices,'uint8');

for slice=1:slices
    seg_img(:,:,slice)=imread(seg_file,slice);
end

bw_seg_img=zeros(rows,cols,slices,'uint8');
bw_seg_img(seg_img~=0)=1;
CC_seg=bwconncomp(bw_seg_img);
fprintf('Total number of blobs: %d\n',CC_seg.NumObjects);

% Do image closing and filling on synapses and junks separately
fprintf('Image closing and filling\n');
im_synapse=zeros(rows,cols,slices,'uint8');
im_synapse(seg_img==255)=1;
im_junk=zeros(rows,cols,slices,'uint8');
im_junk(seg_img==127)=1;
SE=strel('sphere',3);
synapse_closed=imclose(im_synapse,SE);
synapse_closed_filled=imfill(synapse_closed,'holes');
junk_closed=imclose(im_junk,SE);
junk_closed_filled=imfill(junk_closed,'holes');

% For each blob, detetmine if it's a synapse or junk based on voting
fprintf('Now voting on each blob\n');
voted_img=zeros(rows,cols,slices,'uint8');
for obj=1:CC_seg.NumObjects
    
    obj_idx=CC_seg.PixelIdxList{obj};
    % If the blob size is less than 250 voxels, remove from the result
    obj_size=length(obj_idx);
    if obj_size<250
        voted_img(obj_idx)=0; % set as background
        continue;
    end
    
%     % Simple voting based on number of voxels
%     if length(find(seg_img(obj_idx)==255)) >= length(find(seg_img(obj_idx)==127))
%         voted_img(obj_idx)=255;
%     else
%         voted_img(obj_idx)=127;
%     end
    
    % Voting after image closing and filling on the current object
    num_synapse=length(find(synapse_closed_filled(obj_idx)==1));
    num_junk=length(find(junk_closed_filled(obj_idx)==1));
    if num_synapse>=num_junk
        voted_img(obj_idx)=255;
    else
        voted_img(obj_idx)=127;
    end
    
end

fprintf('Use voted junks as junks, previously closed and filled synapses as synapses\n'); 
final_img=zeros(rows,cols,slices,'uint8');
final_img(voted_img==127)=127; % Junks are as final junks

% Use previously image closed and filled synapses as final synapses
synapse_mask=zeros(rows,cols,slices,'uint8');
synapse_mask(voted_img==255)=1;
voted_synapse=synapse_mask.*synapse_closed_filled;
voted_synapse_closed_filled=~bwareaopen(~voted_synapse,10);  % fill small holes again

% Remove blobs that are smaller than 250 voxels
CC_synapse=bwconncomp(voted_synapse_closed_filled);
for obj=1:CC_synapse.NumObjects
    obj_idx=CC_synapse.PixelIdxList{obj};
    if length(obj_idx)<250
        voted_synapse_closed_filled(obj_idx)=0;
    end
end
final_img(voted_synapse_closed_filled==1)=255;

end


% Adopted from https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/
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

% % Remove blobs that are smaller than 250 voxels
% CC=bwconncomp(seg_img);
% for obj=1:CC.NumObjects
%     obj_idx=CC.PixelIdxList{obj};
%     if length(obj_idx)<250
%         seg_img(obj_idx)=0;
%     end
% end

seg_img(seg_img~=0)=255;

end