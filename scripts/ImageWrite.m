function ImageWrite(data,name)

slices=size(data,3);
for slice=1:slices
    if slice==1
        imwrite(data(:,:,slice),name);
    else
        imwrite(data(:,:,slice),name,'WriteMode','append');
    end
end
imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',slices,slices);
t=Tiff(name,'r+');
for slice=1:slices
    setDirectory(t,slice);
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t);