close all;clc;

% read data from disk
dir_work = '/Users/khairyk/Dropbox/data_exchange/image_sample_data/Josh_synapse/';
fn1 = [dir_work '20180503_ch0_2499_2598_crop_01.tif'];
info = imfinfo(fn1);
I = ones(info(1).Height, info(1).Width, numel(info))*255;
for ix = 1:numel(info)
    I(:,:,ix) = imread(fn1,ix);
end

% adjust intensity scaling
I = mat2gray(I, [100 500]);
Io = I;
% background correction (rolling ball)
BG = imopen(I,strel('disk',15));
I2 = I-BG;
% optional: contrast adjustment
I2 = reshape(I2, size(I2,1)*size(I2,2), size(I2,3));
I3 = imadjust(I2);
I3 = reshape(I3, size(I,1), size(I,2), size(I,3));

% optional: median filtering
fac = 7;
I3 = medfilt3(I3, [fac fac fac]);

% show sample images from the volume
indx = 1;
im1 = I(:,:,indx);
im2 = BG(:,:,indx);
im3 = I3(:,:,indx);

% figure(1); cla; imshow(im1);
% figure(2); cla; imshow(im2);
% figure(3); cla; imshow(im3);
% 
% % 
% figure(4); cla; hist(im1(:), 1000);
% figure(5); cla; hist(im2(:), 1000);
% 
% im4 = imsubtract(im1, im2);
% figure(6); cla; hist(im4(:), 1000);
% 
% figure
% surf(double(im2(1:8:end,1:8:end))),zlim([0 1]);
% set(gca,'ydir','reverse');

% figure(7); cla; imshow(im4);
% 
%% save as tif stack
I = mat2gray(I3);
fn_out = [dir_work 'bkrd_subtract.tif'];
imwrite(I(:,:,1), fn_out)
for ix = 1:numel(info)
 imwrite(I(:,:,ix),fn_out,'WriteMode','append')
end
%% binarize image
level = multithresh(I3,3);
seg_I = imquantize(I3, level);
BW = seg_I;
BW(BW>2)=255;
BW(BW<=2) = 0;
C = bwconncomp(BW);
L = labelmatrix(C);
S = regionprops(C);
vol = [S(:).Area];
figure(8);hist(vol/median(vol), 40); title('histogram: relative (to median) volume');%% generate volume histogram
indx = find(vol>500);%% filter for volume
disp(['Number of objects: ' num2str(numel(indx))]);
%% generate surface mesh for objects for visualization
Iobj = zeros(size(I,1), size(I,2), size(I,3));
intensity = zeros(numel(indx),1); % record integrated intensity of this object
for ix = 1:numel(indx)
    Iobj(C.PixelIdxList{indx(ix)}) = 1;
    intensity(ix) = sum(I(C.PixelIdxList{indx(ix)}));
end
figure(9);hist(intensity/median(intensity), 40); title('histogram relative (to median) intensity');%% generate volume histogram

%% save as tif stack
I = mat2gray(Iobj);
fn_out = [dir_work 'lbl.tif'];
imwrite(I(:,:,1), fn_out)
for ix = 1:numel(info)
 imwrite(I(:,:,ix),fn_out,'WriteMode','append')
end
Imasked = Io*0;
Imasked(Iobj==1) = Io(Iobj==1);