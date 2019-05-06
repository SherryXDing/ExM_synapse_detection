delete('compositeImage.tif')
clear all
pathName='/nrs/dickson/lillvis/temp/ExM/opticlobe/L2/20180504/images/stitch/crops_for_Ding_Ackerman/2655-4788-5446/2655-4788-5446_lillvis-manual/';
[~,~,raw]=xlsread([pathName '/266-4788-5446-lillvis-manual-annotation.xlsx']);
r = zeros(600,600,50,'uint16');
g = zeros(600,600,50,'uint16');
for i=2:size(raw,1)
    if strcmp(raw{i,2},'synapse')
        for j=1:50
            g(:,:,j)=g(:,:,j)+imread([pathName '/' num2str(i-2) '.tif'],j);
        end
    else
        for j=1:50
            r(:,:,j)=r(:,:,j)+imread([pathName '/' num2str(i-2) '.tif'],j);
        end
    end
end
for j=1:50
    if j==1
        imwrite(r(:,:,j), 'joshLabledSynapsesAndJunk.tif');
    else
        imwrite(r(:,:,j), 'joshLabledSynapsesAndJunk.tif','writemode','append');
    end
    imwrite(g(:,:,j), 'joshLabledSynapsesAndJunk.tif','writemode','append');
end

imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nchannels=%d\nslices=%d\nhyperstack=true\nmode=composite',100,2,50);
t = Tiff('joshLabledSynapsesAndJunk.tif','r+');
for count=1:100
    setDirectory(t,count)
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t)

g=(g>0);
r=(r>0);
grayscale = uint8(g*255.0+r*127.0);
for j=1:50
    if j==1
        imwrite(grayscale(:,:,j),'C1-[2655, 4788, 5446]-grayscaleLabels.tif');
    else
        imwrite(grayscale(:,:,j),'C1-[2655, 4788, 5446]-grayscaleLabels.tif','writemode','append');
    end
end

imageDescription = sprintf('ImageJ=1.43d\nimages=%d\nslices=%d',50,50);
t = Tiff('C1-[2655, 4788, 5446]-grayscaleLabels.tif','r+');
for count=1:50
    setDirectory(t,count)
    setTag(t,Tiff.TagID.ImageDescription, imageDescription);
    rewriteDirectory(t);
end
close(t)