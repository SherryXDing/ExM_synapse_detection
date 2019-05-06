close all;
clear all;
clc;

% 2655-4788-5446 and 4179-2166-3448
path_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/';
path_4179='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/';
out_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_combine_2655-4179/';

%% BG subtracted data
raw_2655=zeros(600,600,50,'uint16');
for i=1:50
    raw_2655(:,:,i)=imread([path_2655,'C1_2655-4788-5446_BGsubtract.tif'],i);
end
raw_4179=zeros(600,600,50,'uint16');
for i=1:50
    raw_4179(:,:,i)=imread([path_4179,'C2_4179-2166-3448_BGsubtract.tif'],i);
end
[raw_left,raw_right,raw_up,raw_down]=CropAndCombine(raw_2655,raw_4179);

raw_left_name=[out_path,'BGsubtract_left_2655-4179.tif'];
ImageWrite(raw_left,raw_left_name);
raw_right_name=[out_path,'BGsubtract_right_2655-4179.tif'];
ImageWrite(raw_right,raw_right_name);
raw_up_name=[out_path,'BGsubtract_up_2655-4179.tif'];
ImageWrite(raw_up,raw_up_name);
raw_down_name=[out_path,'BGsubtract_down_2655-4179.tif'];
ImageWrite(raw_down,raw_down_name);


%% Mask for headless training (balanced)
mask_2655=zeros(600,600,50,'uint8');
for i=1:50
    mask_2655(:,:,i)=imread([path_2655,'Mask_Headless_Balanced.tif'],i);
end
mask_4179=zeros(600,600,50,'uint8');
for i=1:50
    mask_4179(:,:,i)=imread([path_4179,'Mask_Headless_Balanced_SynapseFused.tif'],i);  % mask_4179(:,:,i)=imread([path_4179,'Mask_Headless_Balanced.tif'],i);
end
[mask_left,mask_right,mask_up,mask_down]=CropAndCombine(mask_2655,mask_4179);

mask_left_name=[out_path,'maskBalanced_SynapseFused_left_2655-4179.tif'];
ImageWrite(mask_left,mask_left_name);
mask_right_name=[out_path,'maskBalanced_SynapseFused_right_2655-4179.tif'];
ImageWrite(mask_right,mask_right_name);
mask_up_name=[out_path,'maskBalanced_SynapseFused_up_2655-4179.tif'];
ImageWrite(mask_up,mask_up_name);
mask_down_name=[out_path,'maskBalanced_SynapseFused_down_2655-4179.tif'];
ImageWrite(mask_down,mask_down_name);


%% Mask labeled with synapse, junk, and edge
label_2655=zeros(600,600,50,'uint8');
for i=1:50
    label_2655(:,:,i)=imread([path_2655,'Mask_Synapse_Junk_Edge.tif'],i);
end
label_4179=zeros(600,600,50,'uint8');
for i=1:50
    label_4179(:,:,i)=imread([path_4179,'Mask_SynapseFused_Junk_Edge.tif'],i);  % label_4179(:,:,i)=imread([path_4179,'Mask_Synapse_Junk_Edge.tif'],i);
end
[label_left,label_right,label_up,label_down]=CropAndCombine(label_2655,label_4179);

label_left_name=[out_path,'Mask_SynapseFused_left_2655-4179.tif'];
ImageWrite(label_left,label_left_name);
label_right_name=[out_path,'Mask_SynapseFused_right_2655-4179.tif'];
ImageWrite(label_right,label_right_name);
label_up_name=[out_path,'Mask_SynapseFused_up_2655-4179.tif'];
ImageWrite(label_up,label_up_name);
label_down_name=[out_path,'Mask_SynapseFused_down_2655-4179.tif'];
ImageWrite(label_down,label_down_name);


%% functions
function [left,right,up,down]=CropAndCombine(image1,image2)
% Input: two images
% Output: combination of left and right, up and down half of two images

[rows,cols,slices]=size(image1);
left=[image1(:,1:cols/2,:),image2(:,1:cols/2,:)];
right=[image1(:,cols/2+1:end,:),image2(:,cols/2+1:end,:)];
up=[image1(1:rows/2,:,:);image2(1:rows/2,:,:)];
down=[image1(rows/2+1:end,:,:);image2(rows/2+1:end,:,:)];

end