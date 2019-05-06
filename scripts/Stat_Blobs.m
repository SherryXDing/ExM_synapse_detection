close all;
clear all;
clc;


%% Size and intensity distribution of synapses and junks in crop 4179-2166-3448
mask_path_4179='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448-lillvis-manual/';
raw_xls_4179='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/4179-2166-3448_annotation_update.xlsx';
% raw_data_4179='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/4179-2166-3448/C2-[4179, 2166, 3448].tif';
raw_data_4179='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_4179-2166-3448/C2_4179-2166-3448_BGsubtract.tif';

[synapse_4179,junk_4179]=CalcProperty(mask_path_4179,raw_xls_4179,raw_data_4179);

figure('Name','Size distribution 4179');
hold on;
histogram(synapse_4179.size,'BinWidth',250);
histogram(junk_4179.size,'BinWidth',250);
legend('Synapse','Junk');
xlabel('Voxels');
ylabel('Count');
title('Size distribution 4179');
hold off;

figure('Name','Mean intensity distribution 4179');
hold on;
histogram(synapse_4179.mean_intensity,'BinWidth',25);
histogram(junk_4179.mean_intensity,'BinWidth',25);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Mean intensity distribution 4179');
hold off;

figure('Name','Total intensity distribution 4179');
hold on;
histogram(synapse_4179.total_intensity,'BinWidth',0.25e6);
histogram(junk_4179.total_intensity,'BinWidth',0.25e6);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Total intensity distribution 4179');
hold off;

figure('Name','Scatter plot 4179');
hold on;
sz=45;
scatter(synapse_4179.mean_intensity,synapse_4179.size,sz,'r^','filled');
scatter(junk_4179.mean_intensity,junk_4179.size,sz,'bo','filled');
legend('Synapse','Junk','Location','northwest');
xlabel('Mean intensity');
ylabel('Size');
title('Blob distribution 4179');
hold off;


%% Size and intensity distribution of synapses and junks in crop 2655-4788-5446
mask_path_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/';
raw_xls_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_annotation_update.xlsx';
% raw_data_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/crops_for_Ding_Ackerman/2655-4788-5446/C1-[2655, 4788, 5446].tif';
raw_data_2655='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/GroundTruth_2655-4788-5446/C1_2655-4788-5446_BGsubtract.tif';

[synapse_2655,junk_2655]=CalcProperty(mask_path_2655,raw_xls_2655,raw_data_2655);

figure('Name','Size distribution 2655');
hold on;
histogram(synapse_2655.size,'BinWidth',250);
histogram(junk_2655.size,'BinWidth',250);
legend('Synapse','Junk');
xlabel('Voxels');
ylabel('Count');
title('Size distribution 2655');
hold off;

figure('Name','Mean intensity distribution 2655');
hold on;
histogram(synapse_2655.mean_intensity,'BinWidth',25);
histogram(junk_2655.mean_intensity,'BinWidth',25);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Mean intensity distribution 2655');
hold off;

figure('Name','Total intensity distribution 2655');
hold on;
histogram(synapse_2655.total_intensity,'BinWidth',0.25e6);
histogram(junk_2655.total_intensity,'BinWidth',0.25e6);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Total intensity distribution 2655');
hold off;

figure('Name','Scatter plot 2655');
hold on;
sz=45;
scatter(synapse_2655.mean_intensity,synapse_2655.size,sz,'r^','filled');
scatter(junk_2655.mean_intensity,junk_2655.size,sz,'bo','filled');
legend('Synapse','Junk','Location','northwest');
xlabel('Mean intensity');
ylabel('Size');
title('Blob distribution 2655');
hold off;


%% Size and intensity distribution of synapses and junks in both crops
figure('Name','Size distribution all');
hold on;
histogram([synapse_2655.size;synapse_4179.size],'BinWidth',250);
histogram([junk_2655.size;junk_4179.size],'BinWidth',250);
legend('Synapse','Junk');
xlabel('Voxels');
ylabel('Count');
title('Size distribution all');
hold off;

figure('Name','Mean intensity distribution all');
hold on;
histogram([synapse_2655.mean_intensity;synapse_4179.mean_intensity],'BinWidth',25);
histogram([junk_2655.mean_intensity;junk_4179.mean_intensity],'BinWidth',25);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Mean intensity distribution all');
hold off;

figure('Name','Total intensity distribution all');
hold on;
histogram([synapse_2655.total_intensity;synapse_4179.total_intensity],'BinWidth',0.25e6);
histogram([junk_2655.total_intensity;junk_2655.total_intensity],'BinWidth',0.25e6);
legend('Synapse','Junk');
xlabel('Intensity');
ylabel('Count');
title('Total intensity distribution all');
hold off;

figure('Name','Scatter plot all');
hold on;
sz=45;
scatter([synapse_2655.mean_intensity;synapse_4179.mean_intensity],[synapse_2655.size;synapse_4179.size],sz,'r^','filled');
scatter([junk_2655.mean_intensity;junk_4179.mean_intensity],[junk_2655.size;junk_4179.size],sz,'bo','filled');
legend('Synapse','Junk','Location','southeast');
xlabel('Mean intensity');
ylabel('Size');
title('Blob distribution all');
hold off;