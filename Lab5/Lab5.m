clc;
clear all;
close all;

%% 2 Chroma Subsampling use grey or colourful?

%% split to Y, Cb and Cr channels - grey
peppers = imread('peppers.png');
peppers = rgb2ycbcr(peppers);
[yImage, crImage, cbImage] = imsplit(peppers);

% Display the original & the various different Y, Cb and Cr
figure;
imshow(yImage, []);
title('Y Image', 'FontSize', 16);
figure;
imshow(cbImage, []);
title('Cb Image', 'FontSize', 16);
figure;
imshow(crImage, []);
title('Cr Image', 'FontSize', 16);

%% split to Y, Cb and Cr channels - colourful like the slides
RGB = imread('peppers.png');
YCBCR = rgb2ycbcr(RGB);

figure;

lb={'Y','Cb','Cr'};
for channel=1:3
    subplot(1,3,channel)
    YCBCR_C=YCBCR;
    YCBCR_C(:,:,setdiff(1:3,channel))=intmax(class(YCBCR_C))/2;
    imshow(ycbcr2rgb(YCBCR_C))
    title([lb{channel} ' component'],'FontSize',18);
    if channel == 1
        Y = YCBCR_C;
    elseif channel == 2
        Cb = YCBCR_C;
    else
        Cr = YCBCR_C;
    end
end

%% Chroma subsampling
cbImage_reduced = imresize(Cb, 1/2);
crImage_reduced = imresize(Cr, 1/2);

cbImage_upsamp = imresize(cbImage_reduced, 2);
crImage_upsamp = imresize(crImage_reduced, 2);

% figure;
% imshow(cbImage_reduced, []);
% title('Cb Image1', 'FontSize', 16);
% figure;
% imshow(crImage_reduced, []);
% title('Cr Image1', 'FontSize', 16);
% 
% figure;
% imshow(cbImage_upsamp, []);
% title('Cb Image2', 'FontSize', 16);
% figure;
% imshow(crImage_upsamp, []);
% title('Cr Image2', 'FontSize', 16);

peppers_recombined = cat(3, Y, Cb, Cr);

figure;
imshow(YCBCR, []);
title('Original');

figure;
imshow(peppers_recombined);
title('Recombined');

%% Luma sub-sampling
yImage_reduced = imresize(yImage, 1/2);
yImage_upsamp = imresize(yImage_reduced, 2);

peppers_recombined_2 = cat(3, yImage_upsamp, cbImage, crImage);
figure;
imshow(peppers_recombined_2, []);
title('Recombined 2');

%% 3 Colour Segmentation - R2021a cannot run this section, only R2021b works. R2021a gives error: Undefined function 'kmeans' for input arguments of type 'double'.
f = imread("peppers.png");
% Create a color transformation structure that defines an sRGB to L*a*b* conversion.
C = makecform('srgb2lab');
% Perform the transformation by using the applycform function.
im_lab = applycform(f,C);

% First reshape the a∗ and b∗ channels:
ab = double(im_lab(:,:,2:3));  % NOT im2double
m = size(ab,1);
n = size(ab,2);
ab = reshape(ab,m*n,2);

% K = 2;
% row = [55   200];
% col = [155   400];
K = 4;
row = [55    130   200  280];
col = [155   110   400  470];
% Convert (r,c) indexing to 1D linear indexing.
idx = sub2ind([size(f,1) size(f,2)], row, col);
% Vectorize starting coordinates
for k = 1:K
  mu(k,:) = ab(idx(k),:);
end

cluster_idx = kmeans(ab, K, 'Start', mu);

% Label each pixel according to k-means
pixel_labels = reshape(cluster_idx, m, n);
h = figure,imshow(pixel_labels, [])
title('Image labeled by cluster index, K=2');
colormap('jet')

% output = repmat(pixel_labels,4);
% mask1 = pixel_labels==1;
% cluster1 = f .* uint8(mask1);
% figure
% imshow(cluster1);
% title('Objects in Cluster 1');

% mask2 = pixel_labels==2;
% cluster2 = f .* uint8(mask2);
% figure
% imshow(cluster2);
% title('Objects in Cluster 2');

% mask3 = pixel_labels==3;
% cluster3 = f .* uint8(mask3);
% figure
% imshow(cluster3);
% title('Objects in Cluster 3');

% mask4 = pixel_labels==4;
% cluster4 = f .* uint8(mask4);
% figure
% imshow(cluster4);
% title('Objects in Cluster 4');