function flag = Postprocessing(img_name, individual_mask, remove_thres)
% Args:
% img_name: name and directory of an image
% individual_mask (optional): if save individual masks (true or false), default is false
% remove_thres (optional): voxels of small piece to remove, default is 10
switch nargin
    case 1
        individual_mask = false;
        remove_thres = 10;
    case 2
        if ~islogical(individual_mask)
            error('individual_mask should be a logical input: true or false');
        end
        remove_thres = 10;
    case 3
        if ~islogical(individual_mask)
            error('individual_mask should be a logical input: true or false');
        end
        if remove_thres <= 0
            error('remove_thres should be larger than 0');
        end
    otherwise
        error('At least one argument, but no more than 3 arguments!');
end
tic;
disp('################################');
[data_path, name, ext] = fileparts(img_name);
disp(['Postprocessing on ', name, '.tif']);
img = read_tif(img_name);
img_close = closing_img(img);
img_close_watershed = water_shed(img_close);
img_close_watershed = remove_small_piece(data_path, img_close_watershed, individual_mask, remove_thres);
write_tif(img_close_watershed, [data_path, '/postprocessed_', name, '.tif']);
toc;
disp('########### Finish #############');
flag=1;
end


function img = read_tif(img_file)
% Read tif image
im_info = imfinfo(img_file);
rows = im_info(1).Height;
cols = im_info(1).Width;
slices = numel(im_info);
img = zeros(rows, cols, slices, 'uint16');
for slice = 1:slices
    img(:,:,slice)=imread(img_file, slice);
end
end


function write_tif(data, name)
% Write data into tif image
slices = size(data, 3);
for slice = 1:slices
    if slice == 1
        imwrite(data(:,:,slice), name);
    else
        imwrite(data(:,:,slice), name, 'WriteMode', 'append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',slices,slices);
t = Tiff(name,'r+');
for slice = 1:slices
    setDirectory(t, slice);
    setTag(t, Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);
end


function seg_img = closing_img(img)
disp('Image closing...')
img(img~=0)=1;
SE=strel('sphere',3);
seg_img=imclose(img,SE);
% % image filling
% seg_img=imfill(seg_img,'holes');
seg_img(seg_img~=0) = 255;
end


function seg_img = water_shed(bw)
% Watershed segmentation
disp('Watershed segmentation...');
bw(bw~=0) = 1;
D = -bwdist(~bw);
mask = imextendedmin(D,2);
D2 = imimposemin(D,mask);
Ld = watershed(D2);
seg_img = bw;
seg_img(Ld==0) = 0;
seg_img(seg_img~=0) = 255;
end


function img = remove_small_piece(data_path, img, remove_thres, individual_mask)
% Remove blobs that are smaller than N voxels
disp('Remove small pieces...');
if individual_mask && exist([data_path, '/individual_masks/'], 'dir') ~= 7
    status = mkdir([data_path, '/individual_masks']);
end
  
CC = bwconncomp(img);
region_prop = regionprops(CC);
centroids = round(cat(1, region_prop.Centroid));
obj_stat = [];
for obj = 1:CC.NumObjects
    obj_idx = CC.PixelIdxList{obj};
    if length(obj_idx) < remove_thres
        img(obj_idx)=0;
    else
        curr_stat = [length(obj_idx), centroids(obj,:)];
        obj_stat = [obj_stat; curr_stat];
        if individual_mask
            curr_mask = zeros(size(img), 'uint16');
            curr_mask(obj_idx) = 255;
            write_tif(curr_mask, [data_path, '/individual_masks/', int2str(obj), '.tif']);
        end
    end
end
xlswrite([data_path, '/synapse_stats.xlsx'], obj_stat);
img(img~=0) = 255;
end