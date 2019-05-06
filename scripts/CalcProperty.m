function [property_synapse,property_junk]=CalcProperty(mask_path,xls_path,raw_data)
% Inputs:
% mask_path: folder in which manually marked masks locate
% xls_path: excel table path that stores the label
% raw_data: raw data path
% Outputs:
% property_synapse, property_junk: a structure indicating size, mean intensity, and total intensity of synapse and junk

[~,~,raw_xls]=xlsread(xls_path);

info=imfinfo(raw_data);
raw_im=zeros(info(1).Height,info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    raw_im(:,:,slice)=imread(raw_data,slice);
end

size_synapse=[];
size_junk=[];
mu_intensity_synapse=[];
mu_intensity_junk=[];
sum_intensity_synapse=[];
sum_intensity_junk=[];

for i=2:size(raw_xls,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    
    im=zeros(info(1).Height,info(1).Width,numel(info),'uint16');
    for slice=1:numel(info)
        im(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice);
    end
    idx=find(im(:)~=0);
    num_voxel=length(idx);
    mu_intensity=mean(raw_im(idx));
    sum_intensity=sum(raw_im(idx));
    if strcmp(raw_xls{i,2},'synapse')
        size_synapse=[size_synapse;num_voxel];
        mu_intensity_synapse=[mu_intensity_synapse;mu_intensity];
        sum_intensity_synapse=[sum_intensity_synapse;sum_intensity];
    elseif strcmp(raw_xls{i,2},'junk')
        size_junk=[size_junk;num_voxel];
        mu_intensity_junk=[mu_intensity_junk;mu_intensity];
        sum_intensity_junk=[sum_intensity_junk;sum_intensity];
    end
end

property_synapse.size=size_synapse;
property_synapse.mean_intensity=mu_intensity_synapse;
property_synapse.total_intensity=sum_intensity_synapse;
property_junk.size=size_junk;
property_junk.mean_intensity=mu_intensity_junk;
property_junk.total_intensity=sum_intensity_junk;