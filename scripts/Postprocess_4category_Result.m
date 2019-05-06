close all;
clear all;
clc;

result_path='/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_combine_2655-4179/';
info=imfinfo([result_path,'Pxl_4category_SynFused_TrainLeft_TestRight.tiff']);
rows=info(1).Height;
cols=info(1).Width;
slices=numel(info);
result=zeros(rows,cols,slices,'uint8');
for slice=1:slices
    result(:,:,slice)=imread([result_path,'Pxl_4category_SynFused_TrainLeft_TestRight.tiff'],slice);
end

result(result==1)=0;
result(result==2)=127;
result(result==3)=255;
result(result==4)=127;
result(result==5)=255;

final_name=[result_path,'Pxl_4category_SynFused_TrainLeft_TestRight_0-255.tif'];
for slice=1:slices
    if slice==1
        imwrite(result(:,:,slice),final_name);
    else
        imwrite(result(:,:,slice),final_name,'WriteMode','append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t=Tiff(final_name,'r+');
for slice=1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);