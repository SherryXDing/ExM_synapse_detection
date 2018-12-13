directory='/groups/scicompsoft/home/ackermand/lillvis/JoshGroundTruth/';
fileNames ={'lbl_withjunk.h5',...
    'Test_Px2cls_Obj_0726.h5',...
    'Test_Px3cls_Obj_0726.h5',...
    'C1-[2655, 4788, 5446]-rightHalf_pixelClassification_Simple Segmentation.h5',...
    'C1-[2655, 4788, 5446]-rightHalf-8bit_autocontext_Simple Segmentation Stage 2.h5',...
    'Test_Px2clsGTtrain_Obj_0726.h5',...
    'Test_Px2cls_Obj_HighSigma_0727.h5',...
    'Test_Px3cls_Obj_HighSigma_0727.h5',...
    'C1-[2655, 4788, 5446]-rightHalf-8bit_autocontext_3classes_Simple Segmentation Stage 2.h5'};
allCombinedObjects=zeros(600,300,50);
% % % for i=1:numel(fileNames)
% % %     segmentedResultsFile=permute(squeeze(h5read([directory fileNames{i}],'/exported_data')),[2 1 3]);
% % %     allCombinedObjects=allCombinedObjects+(segmentedResultsFile>0);
% % % end
% % % allCombinedObjectsLabeled=bwlabeln(allCombinedObjects);


pathName='/nrs/dickson/lillvis/temp/ExM/opticlobe/L2/20180504/images/stitch/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/';
[~,~,raw]=xlsread([pathName '/266-4788-5446-lillvis-manual-annotation.xlsx']);
synapsesWithUIDImage = zeros(600,300,50,'uint8');
junkWithUIDImage = zeros(600,300,50,'uint8');
currentPiece = zeros(600,600,50,'uint16');
synapseLabelCount=128;
junkLabelCount=1;
testingForOverlaps=zeros(600,600,50);
for i=2:size(raw,1)
    for j=1:50
        currentPiece(:,:,j)=imread([pathName '/' num2str(i-2) '.tif'],j);
    end
    indicesOfCurrentPiece = find(currentPiece>0);
    if any(testingForOverlaps(indicesOfCurrentPiece))
        fprintf('%d ',i-2);
        uids = unique(nonzeros(testingForOverlaps(indicesOfCurrentPiece)));
        for u=1:numel(uids)
            fprintf('%d ', uids(u)-2);
        end
        fprintf('\n');
    end
    testingForOverlaps(indicesOfCurrentPiece)=i;
    
    currentPieceCropped=currentPiece(:,301:600,:);
    if strcmp(raw{i,2},'synapse')
        currentPieceCropped(currentPieceCropped>0)=synapseLabelCount;
        synapsesWithUIDImage(currentPieceCropped>0) = uint8(currentPieceCropped(currentPieceCropped>0));
       % fprintf('%d %d %d \n',i,synapseLabelCount, max(synapsesWithUIDImage(:)))
        synapseLabelCount=synapseLabelCount+1;
    else
        currentPieceCropped(currentPieceCropped>0)=junkLabelCount;
        junkWithUIDImage(currentPieceCropped>0) = uint8(currentPieceCropped(currentPieceCropped>0));
        junkLabelCount=junkLabelCount+1;
    end
end

%groundTruthNegativesAndPositives = junkWithUIDImage+synapsesWithUIDImage;
newJunk=zeros(600,300,50,'uint8');
numJunk=junkLabelCount;
%find new junk, will fuse all pieces that aren't already junk
for i=1:numel(fileNames)
    segmentedResultsFile=permute(squeeze(h5read([directory fileNames{i}],'/exported_data')),[2 1 3]);
    labels=bwlabeln(segmentedResultsFile>0);
    combined=(labels>0)+2*(synapsesWithUIDImage>0)+4*double(junkWithUIDImage>0);
    uniqueLabels = unique(labels);
    for j=2:numel(uniqueLabels)%start at 1, not 0 for background
        if ~(any(combined(labels==uniqueLabels(j))==3) || any(combined(labels==uniqueLabels(j))==5)) %then it is not already called a synapse or junk in ground truth
            newJunk(labels==uniqueLabels(j))=1;
      end
  end
end
newJunkLabels=bwlabeln(newJunk);
newJunkLabels(newJunkLabels>0)=newJunkLabels(newJunkLabels>0)+junkLabelCount-1;
groundTruthNegativesAndPositives = junkWithUIDImage+synapsesWithUIDImage+uint8(newJunkLabels);
delete('groundTruthNegativesAndPositives.h5');
h5create('groundTruthNegativesAndPositives.h5','/exported_data',[300,600,50]);
h5write('groundTruthNegativesAndPositives.h5','/exported_data',permute(groundTruthNegativesAndPositives,[2,1,3]));
