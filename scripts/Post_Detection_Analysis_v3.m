close all;
clear all;
clc;

ResultPath='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_combine_2655-4179/Feature_Test/';
% GroundTruthR='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_combine_2655-4179/Mask_right_2655-4179.tif';
GroundTruthR='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_combine_2655-4179/Mask_SynapseFused_right_2655-4179.tif';

% Sigma03R=[ResultPath,'Pxl_Sigma03_TrainLeft_TestRight.tiff'];
% Sigma07R=[ResultPath,'Pxl_Sigma07_TrainLeft_TestRight.tiff'];
% Sigma10R=[ResultPath,'Pxl_Sigma10_TrainLeft_TestRight.tiff'];
% Sigma16R=[ResultPath,'Pxl_Sigma16_TrainLeft_TestRight.tiff'];
% Sigma35R=[ResultPath,'Pxl_Sigma35_TrainLeft_TestRight.tiff'];
% Sigma50R=[ResultPath,'Pxl_Sigma50_TrainLeft_TestRight.tiff'];
% Sigma100R=[ResultPath,'Pxl_Sigma100_TrainLeft_TestRight.tiff'];
% GaussianSmoothingR=[ResultPath,'Pxl_GaussianSmoothing_TrainLeft_TestRight.tiff'];
% LaplacianOfGaussianR=[ResultPath,'Pxl_LaplacianOfGaussian_TrainLeft_TestRight.tiff'];
% GaussianGradientMagnitudeR=[ResultPath,'Pxl_GaussianGradientMagnitude_TrainLeft_TestRight.tiff'];
% DifferenceOfGaussiansR=[ResultPath,'Pxl_DifferenceOfGaussians_TrainLeft_TestRight.tiff'];
% StructureTensorEigenvaluesR=[ResultPath,'Pxl_StructureTensorEigenvalues_TrainLeft_TestRight.tiff'];
% HessianOfGaussianEigenvaluesR=[ResultPath,'Pxl_HessianOfGaussianEigenvalues_TrainLeft_TestRight.tiff'];
% AllFeaturesR=[ResultPath,'Pxl_All_TrainLeft_TestRight.tiff'];
% SelectedFeatures1R=[ResultPath,'Pxl_SelectedFeatures1_TrainLeft_TestRight.tiff'];
% SelectedFeatures2R=[ResultPath,'Pxl_SelectedFeatures2_TrainLeft_TestRight.tiff'];
PxlProbMapObjR='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_combine_2655-4179/PxlProbMap_Obj_TrainLeft_TestRight.tiff';
PxlSynFusedMskR='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_combine_2655-4179/Pxl_Features2_SynFusedMsk_TrainLeft_TestRight.tiff';
Pxl4categoryR='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_combine_2655-4179/Pxl_4category_SynFused_TrainLeft_TestRight_0-255.tif';

% [PrecisionSigma03RPxl,RecallSigma03RPxl,IouSigma03RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma03R);
% [PrecisionSigma07RPxl,RecallSigma07RPxl,IouSigma07RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma07R);
% [PrecisionSigma10RPxl,RecallSigma10RPxl,IouSigma10RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma10R);
% [PrecisionSigma16RPxl,RecallSigma16RPxl,IouSigma16RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma16R);
% [PrecisionSigma35RPxl,RecallSigma35RPxl,IouSigma35RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma35R);
% [PrecisionSigma50RPxl,RecallSigma50RPxl,IouSigma50RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma50R);
% [PrecisionSigma100RPxl,RecallSigma100RPxl,IouSigma100RPxl]=CalcPrecisionRecallPxl(GroundTruthR,Sigma100R);
% [PrecisionGaussianSmoothingRPxl,RecallGaussianSmoothingRPxl,IouGaussianSmoothingRPxl]=CalcPrecisionRecallPxl(GroundTruthR,GaussianSmoothingR);
% [PrecisionLaplacianOfGaussianRPxl,RecallLaplacianOfGaussianRPxl,IouLaplacianOfGaussianRPxl]=CalcPrecisionRecallPxl(GroundTruthR,LaplacianOfGaussianR);
% [PrecisionGaussianGradientMagnitudeRPxl,RecallGaussianGradientMagnitudeRPxl,IouGaussianGradientMagnitudeRPxl]=CalcPrecisionRecallPxl(GroundTruthR,GaussianGradientMagnitudeR);
% [PrecisionDifferenceOfGaussiansRPxl,RecallDifferenceOfGaussiansRPxl,IouDifferenceOfGaussiansRPxl]=CalcPrecisionRecallPxl(GroundTruthR,DifferenceOfGaussiansR);
% [PrecisionStructureTensorEigenvaluesRPxl,RecallStructureTensorEigenvaluesRPxl,IouStructureTensorEigenvaluesRPxl]=CalcPrecisionRecallPxl(GroundTruthR,StructureTensorEigenvaluesR);
% [PrecisionHessianOfGaussianEigenvaluesRPxl,RecallHessianOfGaussianEigenvaluesRPxl,IouHessianOfGaussianEigenvaluesRPxl]=CalcPrecisionRecallPxl(GroundTruthR,HessianOfGaussianEigenvaluesR);
% [PrecisionAllFeaturesRPxl,RecallAllFeaturesRPxl,IouAllFeaturesRPxl]=CalcPrecisionRecallPxl(GroundTruthR,AllFeaturesR);
% [PrecisionSelectedFeatures1RPxl,RecallSelectedFeatures1RPxl,IouSelectedFeatures1RPxl]=CalcPrecisionRecallPxl(GroundTruthR,SelectedFeatures1R);
% [PrecisionSelectedFeatures2RPxl,RecallSelectedFeatures2RPxl,IouSelectedFeatures2RPxl]=CalcPrecisionRecallPxl(GroundTruthR,SelectedFeatures2R);
[PrecisionProbMapPxl,RecallProbMapPxl,IouProbMapPxl]=CalcPrecisionRecallPxl(GroundTruthR,PxlProbMapObjR);
[PrecisionSynFusedPxl,RecallSynFusedPxl,IouSynFusedPxl]=CalcPrecisionRecallPxl(GroundTruthR,PxlSynFusedMskR);
[Precision4categoryPxl,Recall4categoryPxl,Iou4categoryPxl]=CalcPrecisionRecallPxl(GroundTruthR,Pxl4categoryR);
fprintf('ProbMap_Obj, pixel-level: precision=%f, recall=%f, IoU=%f\n',PrecisionProbMapPxl,RecallProbMapPxl,IouProbMapPxl);
fprintf('SynFused, pixel-level: precision=%f, recall=%f, IoU=%f\n',PrecisionSynFusedPxl,RecallSynFusedPxl,IouSynFusedPxl);
fprintf('4 category, pixel-level: precision=%f, recall=%f, IoU=%f\n',Precision4categoryPxl,Recall4categoryPxl,Iou4categoryPxl);

% fid_pxl=fopen([ResultPath,'Precision_Recall_Pxl.txt'],'w');
% fprintf(fid_pxl,'Method                          | Precision |Recall     |IoU\n');
% fprintf(fid_pxl,'Sigma 0.3                       | %f  | %f  | %f\n',PrecisionSigma03RPxl,RecallSigma03RPxl,IouSigma03RPxl);
% fprintf(fid_pxl,'Sigma 0.7                       | %f  | %f  | %f\n',PrecisionSigma07RPxl,RecallSigma07RPxl,IouSigma07RPxl);
% fprintf(fid_pxl,'Sigma 1.0                       | %f  | %f  | %f\n',PrecisionSigma10RPxl,RecallSigma10RPxl,IouSigma10RPxl);
% fprintf(fid_pxl,'Sigma 1.6                       | %f  | %f  | %f\n',PrecisionSigma16RPxl,RecallSigma16RPxl,IouSigma16RPxl);
% fprintf(fid_pxl,'Sigma 3.5                       | %f  | %f  | %f\n',PrecisionSigma35RPxl,RecallSigma35RPxl,IouSigma35RPxl);
% fprintf(fid_pxl,'Sigma 5.0                       | %f  | %f  | %f\n',PrecisionSigma50RPxl,RecallSigma50RPxl,IouSigma50RPxl);
% fprintf(fid_pxl,'Sigma 10.0                      | %f  | %f  | %f\n',PrecisionSigma100RPxl,RecallSigma100RPxl,IouSigma100RPxl);
% fprintf(fid_pxl,'Gaussian Smoothing              | %f  | %f  | %f\n',PrecisionGaussianSmoothingRPxl,RecallGaussianSmoothingRPxl,IouGaussianSmoothingRPxl);
% fprintf(fid_pxl,'Laplacian Of Gaussian           | %f  | %f  | %f\n',PrecisionLaplacianOfGaussianRPxl,RecallLaplacianOfGaussianRPxl,IouLaplacianOfGaussianRPxl);
% fprintf(fid_pxl,'Gaussian Gradient Magnitude     | %f  | %f  | %f\n',PrecisionGaussianGradientMagnitudeRPxl,RecallGaussianGradientMagnitudeRPxl,IouGaussianGradientMagnitudeRPxl);
% fprintf(fid_pxl,'Difference Of Gaussians         | %f  | %f  | %f\n',PrecisionDifferenceOfGaussiansRPxl,RecallDifferenceOfGaussiansRPxl,IouDifferenceOfGaussiansRPxl);
% fprintf(fid_pxl,'Structure Tensor Eigenvalues    | %f  | %f  | %f\n',PrecisionStructureTensorEigenvaluesRPxl,RecallStructureTensorEigenvaluesRPxl,IouStructureTensorEigenvaluesRPxl);
% fprintf(fid_pxl,'Hessian Of Gaussian Eigenvalues | %f  | %f  | %f\n',PrecisionHessianOfGaussianEigenvaluesRPxl,RecallHessianOfGaussianEigenvaluesRPxl,IouHessianOfGaussianEigenvaluesRPxl);
% fprintf(fid_pxl,'All Features                    | %f  | %f  | %f\n',PrecisionAllFeaturesRPxl,RecallAllFeaturesRPxl,IouAllFeaturesRPxl);
% fprintf(fid_pxl,'Selected Features 1             | %f  | %f  | %f\n',PrecisionSelectedFeatures1RPxl,RecallSelectedFeatures1RPxl,IouSelectedFeatures1RPxl);
% fprintf(fid_pxl,'Selected Features 2             | %f  | %f  | %f\n',PrecisionSelectedFeatures2RPxl,RecallSelectedFeatures2RPxl,IouSelectedFeatures2RPxl);
% fclose(fid_pxl);

% [PrecisionSigma03RObj,RecallSigma03RObj,IouSigma03RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma03R);
% [PrecisionSigma07RObj,RecallSigma07RObj,IouSigma07RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma07R);
% [PrecisionSigma10RObj,RecallSigma10RObj,IouSigma10RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma10R);
% [PrecisionSigma16RObj,RecallSigma16RObj,IouSigma16RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma16R);
% [PrecisionSigma35RObj,RecallSigma35RObj,IouSigma35RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma35R);
% [PrecisionSigma50RObj,RecallSigma50RObj,IouSigma50RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma50R);
% [PrecisionSigma100RObj,RecallSigma100RObj,IouSigma100RObj]=CalcPrecisionRecallObj(GroundTruthR,Sigma100R);
% [PrecisionGaussianSmoothingRObj,RecallGaussianSmoothingRObj,IouGaussianSmoothingRObj]=CalcPrecisionRecallObj(GroundTruthR,GaussianSmoothingR);
% [PrecisionLaplacianOfGaussianRObj,RecallLaplacianOfGaussianRObj,IouLaplacianOfGaussianRObj]=CalcPrecisionRecallObj(GroundTruthR,LaplacianOfGaussianR);
% [PrecisionGaussianGradientMagnitudeRObj,RecallGaussianGradientMagnitudeRObj,IouGaussianGradientMagnitudeRObj]=CalcPrecisionRecallObj(GroundTruthR,GaussianGradientMagnitudeR);
% [PrecisionDifferenceOfGaussiansRObj,RecallDifferenceOfGaussiansRObj,IouDifferenceOfGaussiansRObj]=CalcPrecisionRecallObj(GroundTruthR,DifferenceOfGaussiansR);
% [PrecisionStructureTensorEigenvaluesRObj,RecallStructureTensorEigenvaluesRObj,IouStructureTensorEigenvaluesRObj]=CalcPrecisionRecallObj(GroundTruthR,StructureTensorEigenvaluesR);
% [PrecisionHessianOfGaussianEigenvaluesRObj,RecallHessianOfGaussianEigenvaluesRObj,IouHessianOfGaussianEigenvaluesRObj]=CalcPrecisionRecallObj(GroundTruthR,HessianOfGaussianEigenvaluesR);
% [PrecisionAllFeaturesRObj,RecallAllFeaturesRObj,IouAllFeaturesRObj]=CalcPrecisionRecallObj(GroundTruthR,AllFeaturesR);
% [PrecisionSelectedFeatures1RObj,RecallSelectedFeatures1RObj,IouSelectedFeatures1RObj]=CalcPrecisionRecallObj(GroundTruthR,SelectedFeatures1R);
% [PrecisionSelectedFeatures2RObj,RecallSelectedFeatures2RObj,IouSelectedFeatures2RObj]=CalcPrecisionRecallObj(GroundTruthR,SelectedFeatures2R);
[PrecisionProbMapObj,RecallProbMapObj,IouProbMapObj]=CalcPrecisionRecallObj(GroundTruthR,PxlProbMapObjR);
[PrecisionSynFusedObj,RecallSynFusedObj,IouSynFusedObj]=CalcPrecisionRecallObj(GroundTruthR,PxlSynFusedMskR);
[Precision4categoryObj,Recall4categoryObj,Iou4categoryObj]=CalcPrecisionRecallObj(GroundTruthR,Pxl4categoryR);
fprintf('ProbMap_Obj, obj-level: precision=%f, recall=%f, IoU=%f\n',PrecisionProbMapObj,RecallProbMapObj,IouProbMapObj);
fprintf('SynFused, obj-level: precision=%f, recall=%f, IoU=%f\n',PrecisionSynFusedObj,RecallSynFusedObj,IouSynFusedObj);
fprintf('4 category, obj-level: precision=%f, recall=%f, IoU=%f\n',Precision4categoryObj,Recall4categoryObj,Iou4categoryObj);

% fid_obj=fopen([ResultPath,'Precision_Recall_Obj.txt'],'w');
% fprintf(fid_obj,'Method                          | Precision |Recall     |IoU\n');
% fprintf(fid_obj,'Sigma 0.3                       | %f  | %f  | %f\n',PrecisionSigma03RObj,RecallSigma03RObj,IouSigma03RObj);
% fprintf(fid_obj,'Sigma 0.7                       | %f  | %f  | %f\n',PrecisionSigma07RObj,RecallSigma07RObj,IouSigma07RObj);
% fprintf(fid_obj,'Sigma 1.0                       | %f  | %f  | %f\n',PrecisionSigma10RObj,RecallSigma10RObj,IouSigma10RObj);
% fprintf(fid_obj,'Sigma 1.6                       | %f  | %f  | %f\n',PrecisionSigma16RObj,RecallSigma16RObj,IouSigma16RObj);
% fprintf(fid_obj,'Sigma 3.5                       | %f  | %f  | %f\n',PrecisionSigma35RObj,RecallSigma35RObj,IouSigma35RObj);
% fprintf(fid_obj,'Sigma 5.0                       | %f  | %f  | %f\n',PrecisionSigma50RObj,RecallSigma50RObj,IouSigma50RObj);
% fprintf(fid_obj,'Sigma 10.0                      | %f  | %f  | %f\n',PrecisionSigma100RObj,RecallSigma100RObj,IouSigma100RObj);
% fprintf(fid_obj,'Gaussian Smoothing              | %f  | %f  | %f\n',PrecisionGaussianSmoothingRObj,RecallGaussianSmoothingRObj,IouGaussianSmoothingRObj);
% fprintf(fid_obj,'Laplacian Of Gaussian           | %f  | %f  | %f\n',PrecisionLaplacianOfGaussianRObj,RecallLaplacianOfGaussianRObj,IouLaplacianOfGaussianRObj);
% fprintf(fid_obj,'Gaussian Gradient Magnitude     | %f  | %f  | %f\n',PrecisionGaussianGradientMagnitudeRObj,RecallGaussianGradientMagnitudeRObj,IouGaussianGradientMagnitudeRObj);
% fprintf(fid_obj,'Difference Of Gaussians         | %f  | %f  | %f\n',PrecisionDifferenceOfGaussiansRObj,RecallDifferenceOfGaussiansRObj,IouDifferenceOfGaussiansRObj);
% fprintf(fid_obj,'Structure Tensor Eigenvalues    | %f  | %f  | %f\n',PrecisionStructureTensorEigenvaluesRObj,RecallStructureTensorEigenvaluesRObj,IouStructureTensorEigenvaluesRObj);
% fprintf(fid_obj,'Hessian Of Gaussian Eigenvalues | %f  | %f  | %f\n',PrecisionHessianOfGaussianEigenvaluesRObj,RecallHessianOfGaussianEigenvaluesRObj,IouHessianOfGaussianEigenvaluesRObj);
% fprintf(fid_obj,'All Features                    | %f  | %f  | %f\n',PrecisionAllFeaturesRObj,RecallAllFeaturesRObj,IouAllFeaturesRObj);
% fprintf(fid_obj,'Selected Features 1             | %f  | %f  | %f\n',PrecisionSelectedFeatures1RObj,RecallSelectedFeatures1RObj,IouSelectedFeatures1RObj);
% fprintf(fid_obj,'Selected Features 2             | %f  | %f  | %f\n',PrecisionSelectedFeatures2RObj,RecallSelectedFeatures2RObj,IouSelectedFeatures2RObj);
% fclose(fid_obj);


%% Functions of precision=TP/(TP+FP) and recall=TP/(TP+FN)
% Pixel wise
function [precision,recall, IoU]=CalcPrecisionRecallPxl(ground_truth_file,test_result_file)

TP=0;
FP=0;
FN=0;
im_info=imfinfo(ground_truth_file);
rows=im_info(1).Height;
cols=im_info(1).Width;
slices=numel(im_info);

ground_truth=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    ground_truth(:,:,slice)=imread(ground_truth_file,slice);
end
test_result=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    test_result(:,:,slice)=imread(test_result_file,slice);
end

% If current object is too small, consider it as a junk
bw_test_result=zeros(rows,cols,slices,'uint8');
bw_test_result(test_result~=0)=1;
CC_test=bwconncomp(bw_test_result);
for obj=1:CC_test.NumObjects
    curr_idx=CC_test.PixelIdxList{obj};
    if length(curr_idx)<250
        test_result(curr_idx)=127;
    end
end

% If ground truth is synapse, TP: synapse deteced as synapse, FN: synapse detected as junk
label_synapse=zeros(rows,cols,slices,'uint8');
label_synapse(ground_truth==255)=1;
test_synapse=label_synapse.*test_result;
TP=TP+length(find(test_synapse(:)==255));
FN=FN+length(find(test_synapse(:)==127));

% If ground truth is junk, FP: junk detected as synapse
label_junk=zeros(rows,cols,slices,'uint8');
label_junk(ground_truth==127)=1;
test_junk=label_junk.*test_result;
FP=FP+length(find(test_junk(:)==255));

% If background, FP: background detected as synapse
label_background=zeros(rows,cols,slices,'uint8');
label_background(ground_truth==0)=1;
test_background=label_background.*test_result;
FP=FP+length(find(test_background(:)==255));

precision=TP/(TP+FP);
recall=TP/(TP+FN);
IoU=TP/(TP+FP+FN);

end


% Object wise
function [precision,recall,IoU]=CalcPrecisionRecallObj(ground_truth_file,test_result_file)

% precision=TP/(TP+FP),recall=TP/(TP+FN)

TP=0;
FP=0;
FN=0;
im_info=imfinfo(ground_truth_file);
rows=im_info(1).Height;
cols=im_info(1).Width;
slices=numel(im_info);

ground_truth=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    ground_truth(:,:,slice)=imread(ground_truth_file,slice);
end
test_result=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    test_result(:,:,slice)=imread(test_result_file,slice);
end

bw_test_result=zeros(rows,cols,slices,'uint8');
bw_test_result(test_result~=0)=1;
CC_test=bwconncomp(bw_test_result);

for obj=1:CC_test.NumObjects
    % If current object is too small, consider it as a junk
    curr_idx=CC_test.PixelIdxList{obj};
    if length(curr_idx)<250
        test_result(curr_idx)=127;
    end
end

for obj=1:CC_test.NumObjects
    
    % Whether current test object is a synapse or a junk
    curr_idx=CC_test.PixelIdxList{obj};
    curr_obj=test_result(curr_idx);
    curr_obj_val=0;
    if length(find(curr_obj==255)) >= length(find(curr_obj==127)) 
        curr_obj_val=255; % judge current object as a synapse
    else
        curr_obj_val=127;
    end
    % Whether current ground truth object is a synapse or a junk
    gt_obj=ground_truth(curr_idx);
    gt_obj_val=max(gt_obj);
    
    switch gt_obj_val
        case 255
            if curr_obj_val==255
                TP=TP+1;
            else
                FN=FN+1;
            end
        case 127
            if curr_obj_val==255
                FP=FP+1;
            end
        case 0
            if curr_obj_val==255
                FP=FP+1;
            end
    end
    
end


% If any missed synapse
test_result_bg=uint8(~bw_test_result);
gt_bg=ground_truth.*test_result_bg;
bw_gt_bg=zeros(rows,cols,slices,'uint8');
bw_gt_bg(gt_bg~=0)=1;
CC_gt=bwconncomp(bw_gt_bg);
if CC_gt.NumObjects>0
    for obj=1:CC_gt.NumObjects
        curr_idx=CC_gt.PixelIdxList{obj};
        if max(gt_bg(curr_idx))==255
            FN=FN+1;
        end
    end 
end

precision=TP/(TP+FP);
recall=TP/(TP+FN);
IoU=TP/(TP+FP+FN);
end