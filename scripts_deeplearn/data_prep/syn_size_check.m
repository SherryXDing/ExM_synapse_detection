close all;
clear all;
clc;

data_path = '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/';
path_2655 = [data_path, 'GroundTruth_2655-4788-5446/Mask_SynapseFused_Junk.tif'];
path_4179 = [data_path, 'GroundTruth_4179-2166-3448/Mask_SynapseFused_Junk.tif'];

[CC_2655, regionprop_2655] = SynBoundBox(path_2655);
[CC_4179, regionprop_4179] = SynBoundBox(path_4179);

[maxbox_2655, idx_2655] = MaxBoundBox(regionprop_2655.BoundingBox);
[maxbox_4179, idx_4179] = MaxBoundBox(regionprop_4179.BoundingBox);
max([maxbox_2655; maxbox_4179])

%% Functions
function [CC, region_prop] = SynBoundBox(mask_path)
mask_info = imfinfo(mask_path);
rows = mask_info(1).Height;
cols = mask_info(1).Width;
slices = numel(mask_info);

mask = zeros(rows, cols, slices, 'uint8');
for slice = 1:slices
    mask(:,:,slice) = imread(mask_path, slice);
end
mask(mask==127)=0;
mask(mask==255)=1;

CC = bwconncomp(mask);
region_prop = regionprops('table', CC, 'BoundingBox', 'PixelIdxList');
end


function [max_size, index] = MaxBoundBox(bound_box)
box_size=[];
for i=1:size(bound_box,1)
    curr_box = bound_box(i,:);
    box_size = [box_size; curr_box(4:end)];
end
[max_size, index] = max(box_size);
end