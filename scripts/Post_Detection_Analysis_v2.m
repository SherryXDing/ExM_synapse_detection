close all;
clear all;
clc;

% Josh's labeled ground truth
gt_file='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/JoshGroundTruth/C1-[2655, 4788, 5446]-rightHalf-grayscaleLabels.tif';
info_gt=imfinfo(gt_file);
gt_labeled=zeros(info_gt(1).Height,info_gt(1).Width,numel(info_gt),'uint16');
for slice=1:numel(info_gt)
    gt_labeled(:,:,slice)=imread(gt_file,slice);
end

% Raw data
raw_data='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/JoshGroundTruth/C1-[2655, 4788, 5446].tif';
info_raw=imfinfo(raw_data);
raw_im=zeros(info_raw(1).Height,info_raw(1).Width,numel(info_raw),'uint16');
for slice=1:numel(info_raw)
    raw_im(:,:,slice)=imread(raw_data,slice);
end
raw_im=raw_im(:,0.5*info_raw(1).Width+1:end,:);


%% Generate a mask including Josh labeled blobs and incorrectly detected blobs from testing results
file_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/JoshGroundTruth_output/';
files={'pxlClassification.tif','autoContext.tif',...
    'Test_Px2clsGTtrain_Obj_LowSigma_0726.tiff','Test_Px2cls_Obj_HighSigma_0727.tiff',...
    'Test_Px2cls_Obj_LowSigma_0726.tiff','Test_Px3cls_Obj_HighSigma_0727.tiff',...
    'Test_Px3cls_Obj_LowSigma_0726.tiff'};

mask=zeros(info_gt(1).Height,info_gt(1).Width,numel(info_gt),'uint16');
for i=1:numel(files)
    curr_im=zeros(info_gt(1).Height,info_gt(1).Width,numel(info_gt),'uint16');
    for slice=1:numel(info_gt)
        curr_im(:,:,slice)=imread([file_path,files{i}],slice);
    end
    curr_im(curr_im>0)=1;
    mask=mask+curr_im;
end
mask(mask>0)=1;
mask=mask+gt_labeled;
mask_new_junk=zeros(info_gt(1).Height,info_gt(1).Width,numel(info_gt),'uint16');
[label,num]=bwlabeln(mask);
for i=1:num
    idx=find(label==i);
    if max(mask(idx))==1
        mask_new_junk(idx)=127;
    end
end
mask_final=mask_new_junk+gt_labeled;

% % save as .tif file
% im_out=mat2gray(mask_final);
% fn_out=[file_path,'final_mask.tif'];
% imwrite(im_out(:,:,1),fn_out);
% for slice=1:numel(info_gt)
%     imwrite(im_out(:,:,slice),fn_out,'WriteMode','append');
% end


%% Calculate number of misclassified synapses and junks
mis_syn_all=[];
mis_junk_all=[];
for i=1:numel(files)
    disp(files{i});
    curr_im=zeros(info_gt(1).Height,info_gt(1).Width,numel(info_gt),'uint16');
    for slice=1:numel(info_gt)
        curr_im(:,:,slice)=imread([file_path,files{i}],slice);
    end
    result=CalcMisclassifiedBlob(curr_im,raw_im,mask_final);
    mis_syn_all=[mis_syn_all;result.size_synapse result.mean_intensity_synapse];
    mis_junk_all=[mis_junk_all;result.size_junk result.mean_intensity_junk];
    fprintf('True positive: %d, False positive: %d, True negative: %d, False Negative: %d\n',...
        result.true_positive,result.false_positive,result.true_negative,result.false_negative);
end

% Plot the distribution of misclassified blobs
figure('Name','Scatter plot of misclassified blobs');
hold on;
sz=40;
scatter(mis_syn_all(:,2),mis_syn_all(:,1),sz,'ro','filled');
scatter(mis_junk_all(:,2),mis_junk_all(:,1),sz,'b^','filled');
legend('Synapse','Junk');
xlabel('Mean intensity');
ylabel('Size');
title('Misclassified blobs');


%% functions
function result=CalcMisclassifiedBlob(test_im,raw_im,mask_im)
% Calculate the size and mean intensity of the misclassified blobs
% Definition of misclassified synapse: missed + misclassified as junk
% Definition of miclassified junk: misclassified as synapse
% Input:
% test_im: test result; raw_im: raw data, mask_im: mask image
% Output: result is a structure including
% result.size_synapse: voxel size of synapse that is misclassified (false negative);
% result.mean_intensity_synapse: mean intensity of synapse blob that is misclassified;
% result.size_junk: voxel size of junk that is misclassified (false positive);
% result.mean_intensity: mean intensity of junk blob that is misclassified;
% result.true_positive;
% result.true_negative;
% result.false_positive;
% result.false_negative;

sz_mis_syn=[];
sz_mis_junk=[];
muint_mis_syn=[];
muint_mis_junk=[];
TP=0;
TN=0;
FP=0;
FN=0;

[label,num]=bwlabeln(mask_im);
for i=1:num
    idx=find(label==i);
    mask_val=max(mask_im(idx));
    test_val=max(test_im(idx));
    switch mask_val
        case 255
            if test_val==255
                TP=TP+1;
            else % not detected or mis classified
                FN=FN+1;
                sz_mis_syn=[sz_mis_syn;length(idx)];
                muint_mis_syn=[muint_mis_syn;mean(raw_im(idx))];
            end
        case 127
            if test_val==127 || test_val==0
                TN=TN+1;
            else % misclassified
                FP=FP+1;
                sz_mis_junk=[sz_mis_junk;length(idx)];
                muint_mis_junk=[muint_mis_junk;mean(raw_im(idx))];
            end
        otherwise
            fprintf('No. %d blob in the mask data has an incorrect value\n', i);
    end
end
result.size_synapse=sz_mis_syn;
result.mean_intensity_synapse=muint_mis_syn;
result.size_junk=sz_mis_junk;
result.mean_intensity_junk=muint_mis_junk;
result.true_positive=TP;
result.true_negative=TN;
result.false_positive=FP;
result.false_negative=FN;

end