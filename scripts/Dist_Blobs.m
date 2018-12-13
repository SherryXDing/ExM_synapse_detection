close all;
clear all;
clc;

% Distance between blobs in synapse masks

mask_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448-lillvis-manual/';
mask_info=imfinfo([mask_path,'0.tif']);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);

[~,~,exl]=xlsread('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448_annotation_update.xlsx');
fid=fopen('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/Blobs_Property.txt','w');

mask_all=zeros(rows,cols,slices,'uint8');
dist_within=[];
for i=2:size(exl,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    
    % If mask exists
    fprintf(fid,'\nMask %d\n',i-2);
    mask=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        mask(:,:,slice)=imread([mask_path,int2str(i-2),'.tif'],slice);
    end
    mask(mask~=0)=1;
    if i==163  % Mask 163 has a wrong blob
        L=bwlabeln(mask);
        mask(L==1)=0;
    end
    
    cc_curr=bwconncomp(mask);
    fprintf(fid,'Number of blobs: %d\n',cc_curr.NumObjects);
    for k=1:cc_curr.NumObjects
        fprintf(fid,'Size of blob %d: %d\n',k,length(cc_curr.PixelIdxList{k}));
    end
    
    if cc_curr.NumObjects>1
        centroid_curr=regionprops3(cc_curr,'Centroid');
        d=pdist(table2array(centroid_curr));
        fprintf(fid,'Min distance is: %f\n',max(d));
        fprintf(fid,'Max distance is: %f\n',min(d));
        dist_within=[dist_within d];
    end
    
    mask_all(mask~=0)=1;
end
fprintf(fid,'\n\nMin distance between two blobs within a mask is %f\n', min(dist_within));
fprintf(fid,'Max distance between two blobs within a mask is %f\n', max(dist_within));

cc_all=bwconncomp(mask_all);
centroid_all=regionprops3(cc_all,'Centroid');
dist_all=pdist(table2array(centroid_all));
link_all=linkage(dist_all);
fprintf(fid,'\nMin distance between two blobs in the whole mask is %f\n',min(dist_all));
fprintf(fid,'Max distance between two blobs in the whole mask is %f\n', max(dist_all));
fclose(fid);

figure('Name','Dendrogram');
dendrogram(link_all,size(link_all,1));