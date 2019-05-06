close all;
clear all;
clc;


raw_data='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/JoshGroundTruth/C1-[2655, 4788, 5446].tif';
info=imfinfo(raw_data);
raw_im=zeros(info(1).Height,info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    raw_im(:,:,slice)=imread(raw_data,slice);
end

data_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/JoshGroundTruth_output/';
% Px2cls_obj_HighSigma
cls2_sigmaH=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    cls2_sigmaH(:,:,slice)=imread([data_path,'Test_Px2cls_Obj_HighSigma_0727.tiff'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(cls2_sigmaH,raw_im);
cls2_sigmaH_syn=[sz_mis_syn muint_mis_syn];
cls2_sigmaH_junk=[sz_mis_junk muint_mis_junk];

% Px2cls_obj_LowSigma
cls2_sigmaL=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    cls2_sigmaL(:,:,slice)=imread([data_path,'Test_Px2cls_Obj_LowSigma_0726.tiff'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(cls2_sigmaL,raw_im);
cls2_sigmaL_syn=[sz_mis_syn muint_mis_syn];
cls2_sigmaL_junk=[sz_mis_junk muint_mis_junk];

% Px3cls_obj_HighSigma
cls3_sigmaH=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    cls3_sigmaH(:,:,slice)=imread([data_path,'Test_Px3cls_Obj_HighSigma_0727.tiff'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(cls3_sigmaH,raw_im);
cls3_sigmaH_syn=[sz_mis_syn muint_mis_syn];
cls3_sigmaH_junk=[sz_mis_junk muint_mis_junk];

% Px3cls_obj_LowSigma
cls3_sigmaL=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    cls3_sigmaL(:,:,slice)=imread([data_path,'Test_Px3cls_Obj_LowSigma_0726.tiff'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(cls3_sigmaL,raw_im);
cls3_sigmaL_syn=[sz_mis_syn muint_mis_syn];
cls3_sigmaL_junk=[sz_mis_junk muint_mis_junk];

% Px2cls_GTtrain_Obj_LowSigma
cls2_gt_sigmaL=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    cls2_gt_sigmaL(:,:,slice)=imread([data_path,'Test_Px2clsGTtrain_Obj_LowSigma_0726.tiff'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(cls2_gt_sigmaL,raw_im);
cls2_gt_sigmaL_syn=[sz_mis_syn muint_mis_syn];
cls2_gt_sigmaL_junk=[sz_mis_junk muint_mis_junk];

% Pixcel classification
pxl_classify=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    pxl_classify(:,:,slice)=imread([data_path,'pxlClassification.tif'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(pxl_classify,raw_im);
pxl_classify_syn=[sz_mis_syn muint_mis_syn];
pxl_classify_junk=[sz_mis_junk muint_mis_junk];

% Auto context
auto_context=zeros(info(1).Height,0.5*info(1).Width,numel(info),'uint16');
for slice=1:numel(info)
    auto_context(:,:,slice)=imread([data_path,'autoContext.tif'],slice);
end
[sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(auto_context,raw_im);
auto_context_syn=[sz_mis_syn muint_mis_syn];
auto_context_junk=[sz_mis_junk muint_mis_junk];

[mis_syn_all]=[cls2_sigmaH_syn;cls2_sigmaL_syn;cls3_sigmaH_syn;cls3_sigmaL_syn;cls2_gt_sigmaL_syn;...
    pxl_classify_syn;auto_context_syn];
[mis_junk_all]=[cls2_sigmaH_junk;cls2_sigmaL_junk;cls3_sigmaH_junk;cls3_sigmaL_junk;cls2_gt_sigmaL_junk;...
    pxl_classify_junk;auto_context_junk];

figure('Name','Scatter plot of misclassified');
hold on;
sz=40;
scatter(mis_syn_all(:,2),mis_syn_all(:,1),sz,'ro','filled');
scatter(mis_junk_all(:,2),mis_junk_all(:,1),sz,'b^','filled');
legend('Synapse','Junk');
xlabel('Mean intensity');
ylabel('Size');
title('Misclassified blobs');


%% functions
function [sz_mis_syn,sz_mis_junk,muint_mis_syn,muint_mis_junk]=CalcMisclassified(test_im,raw_im)
% Calculate the size and mean intensity of the misclassified blobs
% Input:
% test_im: test result; raw_im: raw data
% Output:
% sz_mis_syn: voxel size of synapse that is misclassified (false negative)
% sz_junk2syn: voxel size of junk that is misclassified (false positive)
% muint_mis_syn: mean intensity of synapse blob that is misclassified (false negative)
% muint_mis_junk: mean intensity of junk blob that is misclassified (false positive)

msk_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/';
[~,~,label]=xlsread([msk_path,'266-4788-5446-lillvis-manual-annotation.xlsx'],'B2:B123');

sz_mis_syn=[];
sz_mis_junk=[];
muint_mis_syn=[];
muint_mis_junk=[];

[rows,cols,slices]=size(raw_im);
raw_im=raw_im(:,0.5*cols+1:end,:); % right half of the raw image
for blob=1:length(label)
    % read in current mask image
    msk=zeros(rows,cols,slices,'uint16');
    for slice=1:slices
        msk(:,:,slice)=imread([msk_path,num2str(blob-1),'.tif'],slice);
    end
    msk=msk(:,0.5*cols+1:end,:);
    idx=find(msk);
    if isempty(idx)
        continue;
    end
    % the current mask contains blob in the right half
    msk(idx)=1;
    test=test_im.*msk;
    % if not detect the synapse
    if isempty(find(test,1))
        if strcmp(label{blob},'synapse')
            sz_mis_syn=[sz_mis_syn;length(idx)];
            muint_mis_syn=[muint_mis_syn;mean(raw_im(idx))];
%         else
%             sz_mis_junk=[sz_mis_junk;length(idx)];
%             muint_mis_junl=[muint_mis_junk;mean(raw_im(idx))];
        end
        % if misclassified
    else
        val=test(find(test,1));
        % if a synapse is misclassified as a junk
        if strcmp(label{blob},'synapse') && (val~=255)
            sz_mis_syn=[sz_mis_syn;length(idx)];
            muint_mis_syn=[muint_mis_syn;mean(raw_im(idx))];
            % if a junk is misclassified as a synapse
        elseif strcmp(label{blob},'junk') && (val~=127)
            sz_mis_junk=[sz_mis_junk;length(idx)];
            muint_mis_junk=[muint_mis_junk;mean(raw_im(idx))];
        end
    end
end

end