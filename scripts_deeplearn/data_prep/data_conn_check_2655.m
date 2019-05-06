clear all;
close all;
clc;

file_path = '/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/';

%% Check number of connected components of synapses in crop-2655
exl_2655 = [file_path, 'crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_annotation_update.xlsx'];
mask_2655 = [file_path, 'crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/'];
num_conn_comp = [];
[~,~,raw] = xlsread(exl_2655);
mask_info = imfinfo([mask_2655,'0.tif']);
rows = mask_info(1).Height;
cols = mask_info(1).Width;
slices = numel(mask_info);
for i=2:size(raw,1)
    if ~exist([mask_2655,int2str(i-2),'.tif'])
        continue;
    end
    if strcmp(raw{i,2},'synapse')
        mask = zeros(rows,cols,slices,'uint8');
        for slice = 1:slices
            mask(:,:,slice) = imread([mask_2655, num2str(i-2), '.tif'], slice);
        end
        mask(mask~=0) = 1;
        CC = bwconncomp(mask);
        num_conn_comp = [num_conn_comp; i-2, CC.NumObjects];
    end
end

%% If any separate blobs in one synapse mask, fuse it
if max(num_conn_comp(:,2))>1
    idx = find(num_conn_comp(:,2)>1);
    for i=1:length(idx)
        fprintf('Now processing mask %d\n',num_conn_comp(idx(i),1));
        mask = zeros(rows,cols,slices,'uint8');
        for slice = 1:slices
            mask(:,:,slice) = imread([file_path, 'crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/', num2str(num_conn_comp(idx(i),1)), '.tif'], slice);
        end
        r = 4;
        num_obj = num_conn_comp(idx(i),2);
        while num_obj>1
            SE = strel('sphere',r);
            mask_closed = imclose(mask,SE);
            cc = bwconncomp(mask_closed);
            num_obj = cc.NumObjects;
            r=r+2;
        end
        name = [file_path, 'crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/', num2str(num_conn_comp(idx(i),1)), '_fused.tif'];
        ImageWrite(mask_closed, name);
    end
end