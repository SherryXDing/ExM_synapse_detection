close all;
clear all;
clc;

% mask_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448-lillvis-manual/';
% [~,~,raw]=xlsread('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448_annotation_update.xlsx');
% out_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/';
% bgsub_name=[out_path,'C2_4179-2166-3448_BGsubtract.tif'];

mask_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/';
[~,~,raw]=xlsread('/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_annotation_update.xlsx');
out_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/';
bgsub_name=[out_path,'C1_2655-4788-5446_BGsubtract.tif'];

% if exist([out_path,'mask_all.tif']) 
%     delete([out_path,'mask_all.tif']);
% end
% if exist([out_path,'mask_headless.tif'])
%     delete([out_path,'mask_headless.tif']);
% end
% if exist([out_path,'mask_headless_balanced.tif'])
%     delete([out_path,'mask_headless_balanced.tif']);
% end

SynapseJunkMask(mask_path,raw,out_path);
SynapseJunkEdgeMask(mask_path,raw,out_path);
% MaskHeadless(mask_path,raw,out_path,bgsub_name);
% BlobMask(mask_path,raw,out_path);

%% Create a labeled mask with labeled synapse=255,junk=127,others=0
function SynapseJunkMask(mask_path,raw,out_path)

mask_info=imfinfo([mask_path,'0.tif']);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);

mask_synapse_junk=zeros(rows,cols,slices,'uint8');
for i=2:size(raw,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    msk=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice); 
    end
    if strcmp(raw{i,2},'synapse')
        mask_synapse_junk(msk~=0)=255;
    elseif strcmp(raw{i,2},'junk')
        mask_synapse_junk(msk~=0)=127;
    end
end

final_mask_name=[out_path,'Mask_Synapse_Junk.tif'];
ImageWrite(mask_synapse_junk,final_mask_name);

end


%% Create a labeled mask with labeled synapse=255,junk=127,edge=63,others=0
function SynapseJunkEdgeMask(mask_path,raw,out_path)

mask_info=imfinfo([mask_path,'0.tif']);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);

mask_synapse_junk_edge=zeros(rows,cols,slices,'uint8');
for i=2:size(raw,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    msk=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice); 
    end
    switch raw{i,2}
        case 'synapse'
            mask_synapse_junk_edge(msk~=0)=255;
        case 'junk'
            mask_synapse_junk_edge(msk~=0)=127;
        case 'edge'
            mask_synapse_junk_edge(msk~=0)=63;
    end

end

final_mask_name=[out_path,'Mask_Synapse_Junk_Edge.tif'];
ImageWrite(mask_synapse_junk_edge,final_mask_name);

end


%% Create labeled masks (balanced and unbalanced numbers of synapses and junks) with synapse=3,junk=2,background=1,unlabeled=0
function MaskHeadless(mask_path,raw,out_path,bgsub_name)

mask_info=imfinfo([mask_path,'0.tif']);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);
% load a BG subtracted image
bgsub=zeros(rows,cols,slices,'uint16');
for slice=1:slices
    bgsub(:,:,slice)=imread(bgsub_name,slice);
end

% level=multithresh(bgsub,5);
% mask_headless=imquantize(bgsub,level);
% mask_headless(mask_headless>1)=0;  % all blobs
% mask_headless(mask_headless<=1)=1; % background
% mask_headless=uint8(mask_headless);
mask_headless=zeros(rows,cols,slices,'uint8');
mask_headless(bgsub>10)=0;  % all blobs
mask_headless(bgsub<=10)=1; % background

for i=2:size(raw,1)
    if ~exist([mask_path,num2str(i-2),'.tif'])
        continue;
    end
    msk=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice); 
    end
    if strcmp(raw{i,2},'synapse')
        mask_headless(msk~=0)=3;  % synapse
    elseif strcmp(raw{i,2},'junk')
        mask_headless(msk~=0)=2;  % junk
    end
end
% randomly select background
idx=find(mask_headless==1);
idx_rand=randperm(length(idx));
idx_set0=idx_rand(1:round(0.998*length(idx)));
mask_headless(idx(idx_set0))=0;

mask_headless_name=[out_path,'Mask_Headless.tif'];
ImageWrite(mask_headless,mask_headless_name);


% Mask with almost balanced number of synapses and junks
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
mask_headless_balanced=mask_headless;
n_synapse=0;
for i=2:size(raw,1)
    if ~exist([mask_path,num2str(i-2),'.tif'])
        continue;
    end
    if strcmp(raw{i,2},'synapse')
        n_synapse=n_synapse+1;
        if mod(n_synapse,step)~=0  % only keep synapse of every "step" number, set others to 0 (unmarked)
            msk=zeros(rows,cols,slices,'uint16');
            for slice=1:slices
                msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice);
            end
            mask_headless_balanced(msk~=0)=0;  % synapse
        end
    end
end

mask_headless_balanced_name=[out_path,'Mask_Headless_Balanced.tif'];
ImageWrite(mask_headless_balanced,mask_headless_balanced_name);

end


%% Create a mask with marked regions in indices corresponding to the annotation excel form
function BlobMask(mask_path,raw,out_path)

mask_info=imfinfo([mask_path,'0.tif']);
rows=mask_info(1).Height;
cols=mask_info(1).Width;
slices=numel(mask_info);

mask_blob=zeros(rows,cols,slices,'uint8');
for i=2:size(raw,1)
    if ~exist([mask_path,int2str(i-2),'.tif'])
        continue;
    end
    msk=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        msk(:,:,slice)=imread([mask_path,num2str(i-2),'.tif'],slice);
    end
    if strcmp(raw{i,2},'synapse') || strcmp(raw{i,2},'junk') || strcmp(raw{i,2},'edge')
        mask_blob(msk~=0)=i;
    end
    
end

final_mask_name=[out_path,'Mask_Blobs.tif'];
ImageWrite(mask_blob,final_mask_name);

end