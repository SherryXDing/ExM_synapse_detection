close all;
clear all;

%% Get segmented image files
segmentedResultFolder = '/groups/scicompsoft/home/ackermand/lillvis/forIlastik/dataSubset1000_1049/Segmentation/';
segmentedResultFiles = dir([segmentedResultFolder '/*.tiff']);
slices=numel(segmentedResultFiles);
[rows, cols] = size(imread([segmentedResultFiles(1).folder '/' segmentedResultFiles(1).name]));
im3D = zeros(rows,cols,slices); % Will contain the 3D segmented stack
for slice=1:slices
    im=imread([segmentedResultFiles(slice).folder '/' segmentedResultFiles(slice).name]);
    im3D(:,:,slice)=im;
    red=zeros(size(im));
    green=zeros(size(im));
    blue=zeros(size(im));
    % Group subcategories into parent categories
    red(im==1530)=1; % Background 
    blue(im<=765)=1; % Synapse
    green(im>765 & im<1530)=1; % Junk
    comboImage = cat(3,red, green, blue);
    if slice==1
        imwrite(comboImage, 'segmentationChannels.tif'); % Write original segmented results into single, colored file
    else
        imwrite(comboImage, 'segmentationChannels.tif','writemode','append');
    end
end

%% Perform image closing, filling and find connected components
% Just Synapse
synapse3D=zeros(size(im3D));
synapse3D(im3D<=765)=1; 
SE = strel('sphere',4);
synapse3D_closed = imclose(synapse3D, SE);
synapse3D_closed_filled = imfill(synapse3D_closed,'holes');
synapse_connected_components=bwconncomp(synapse3D_closed_filled); 

% Just Junk
junk3D = zeros(size(im3D));
junk3D(im3D>765 & im3D<1530)=1;
junk3D_closed = imclose(junk3D, SE);
junk3D_closed_filled = imfill(junk3D_closed,'holes');
junk_connected_components = bwconncomp(junk3D_closed_filled);

% Plot a histogram of the size distributions of synapses and junk and use
% that to determine a cutoff size
figure();
x=(0:0.25:5);
histogram(log10(cellfun('length',synapse_connected_components.PixelIdxList)),x)
hold on
histogram(log10(cellfun('length',junk_connected_components.PixelIdxList)),x)
xticks([0:1:5]);
xticklabels(10.^(0:1:5))
legend('Synapse','Junk')
xlabel('Voxels');
ylabel('Count');
title('Size Distributions');

cutoffSize=350; %Remove all connected components smaller than this size
synapse3D_final=bwareaopen(synapse3D_closed_filled,cutoffSize); 
junkDeterminedByCutoff=synapse3D_closed_filled-synapse3D_final; %These are groups of previously classified "synapse" pixels that are considered too small to be synapses and so are reclassified as junk

junk3D_combined=max(junk3D,junkDeterminedByCutoff); %Combine the junk points from before with the newly assigned junk and perform closing/filling
junk3D_combined_closed = imclose(junk3D_combined, SE);
junk3D_combined_closed_filled = imfill(junk3D_combined_closed,'holes');
junk3D_final=max(0,junk3D_combined_closed_filled-synapse3D_final); %now junk3D_final only has junk and excludes anything that we already determined is a synapse

synapse_labels=bwlabeln(synapse3D_final); %Assigning each connected component a different label
allSynapses=unique(synapse_labels(synapse_labels>0)); %A list of all the unique synapses (exlcuding background (0))

%% Determine how ground truth synapse points were segmented
ground_truth_synapse_xyz=csvread('/groups/scicompsoft/home/ackermand/lillvis/forIlastik/dataSubset1000_1049/ground_truth.csv');
counts=zeros(3,1);
for slice=0:slices-1
    indices=find(ground_truth_synapse_xyz(:,3)==slice+1);
    for j=indices'
       value =  synapse3D_final(round(ground_truth_synapse_xyz(j,2)), round(ground_truth_synapse_xyz(j,1)), ground_truth_synapse_xyz(j,3))*2 ...
           +junk3D_final(round(ground_truth_synapse_xyz(j,2)), round(ground_truth_synapse_xyz(j,1)), ground_truth_synapse_xyz(j,3))+1;
       % A value of 1 means the ground truth point was segmented as
       % background, a value of 2 means it was junk and a value of 3 means
       % it was synapse
       counts(value)=counts(value)+1;
    end
end

%Plot a histogram showing how ground truth points were classified
figure();
bar(counts)
xticks(1:3);
xticklabels({'Background', 'Junk', 'Synapse'})
title('False Negatives and True Positives')
ylabel('Number of Ground Truth Points');
xlabel('Classification of Ground Truth Synapses');

%Determine which synapses correspond to those marked in the ground truth
synapsesFound=[];
for i=1:size(ground_truth_synapse_xyz,1)
    synapseId=synapse_labels(round(ground_truth_synapse_xyz(i,2)), round(ground_truth_synapse_xyz(i,1)), ground_truth_synapse_xyz(i,3));
    if synapseId~=0  %then not background
     synapsesFound=[synapsesFound; synapseId];
    end
end
synapsesFound=unique(synapsesFound);
% Calculate false positives as all synapses found by segmentation but
% weren't classified as synapses in the ground truth
falsePositives=[];
falsePositives.id = setdiff(allSynapses, synapsesFound);
falsePositives.count = zeros(size(falsePositives.id));

%%  Create output images, where analysisImage contains true synapses in green, ground truth points centered in a square (for easier visibility) in red, and false postive synapses in blue. compositeImage has the original data as red, synapses as green and junk as blue.
analysisImageRed=zeros(size(synapse_labels));
analysisImageBlue=zeros(size(synapse_labels));
analysisImageGreen=zeros(size(synapse_labels));
for i=1:numel(falsePositives.count)
    falsePositives.count(i)=numel(find(synapse_labels==falsePositives.id(i)));
    analysisImageBlue(synapse_labels==falsePositives.id(i))=falsePositives.id(i)+1;
end
for i=synapsesFound'
   analysisImageGreen(synapse_labels==i)=i+1;
end
for i=1:size(ground_truth_synapse_xyz,1)
   analysisImageRed(round(ground_truth_synapse_xyz(i,2))+(-2:2), round(ground_truth_synapse_xyz(i,1))+(-2:2), ground_truth_synapse_xyz(i,3))=255;
end
for i=1:slices
    comboImage = cat(3,analysisImageRed(:,:,i)>1, analysisImageGreen(:,:,i)>1, analysisImageBlue(:,:,i)>1);
    comboImage=comboImage*255.0/255.0;
    originalIm=imread('/groups/scicompsoft/home/ackermand/lillvis/forIlastik/cropped_1000_1049.tif',i);
    if i==1
        imwrite(comboImage, 'analysisImage.tif');
        imwrite(originalIm,'compositeImage.tif');
    else
        imwrite(comboImage, 'analysisImage.tif','writemode','append');
        imwrite(originalIm,'compositeImage.tif','writemode','append');
    end
    %imwrite(uint16(max(analysisImageBlue(:,:,i)>0,analysisImageGreen(:,:,i)>0)*2^16*.5-1),'compositeImage.tif','writemode','append');
    imwrite(uint16(synapse3D_final(:,:,i)*2^16*.5-1),'compositeImage.tif','writemode','append');
    imwrite(uint16(junk3D_final(:,:,i)*2^16*.5-1),'compositeImage.tif','writemode','append');

end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nchannels=%d\nslices=%d\nhyperstack=true\nmode=composite',3*slices,3,slices);
t = Tiff('compositeImage.tif','r+');
for count=1:3*slices
    setDirectory(t,count)
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t)