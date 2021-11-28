clc;
clear all;
close all;

%% split to Y, Cb and Cr channels
RGB = imread('peppers.png');
YCBCR = rgb2ycbcr(RGB);
[Y, Cb, Cr] = imsplit(YCBCR);

% Display the original & the various different Y, Cb and Cr
figure;
imshow(Y, []);
title('Y Image', 'FontSize', 16);
figure;
imshow(Cb, []);
title('Cb Image', 'FontSize', 16);
figure;
imshow(Cr, []);
title('Cr Image', 'FontSize', 16);

% %% split to Y, Cb and Cr channels - colourful like the slides
% RGB = imread('peppers.png');
% YCBCR = rgb2ycbcr(RGB);
% 
% figure;
% 
% lb={'Y','Cb','Cr'};
% for channel=1:3
%     subplot(1,3,channel)
%     YCBCR_C=YCBCR;
%     YCBCR_C(:,:,setdiff(1:3,channel))=intmax(class(YCBCR_C))/2;
%     imshow(ycbcr2rgb(YCBCR_C))
%     title([lb{channel} ' component'],'FontSize',18);
%     if channel == 1
%         Y = YCBCR_C;
%     elseif channel == 2
%         Cb = YCBCR_C;
%     else
%         Cr = YCBCR_C;
%     end
% end

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

recombined = cat(3, Y, Cb, Cr);
recombined_RGB = ycbcr2rgb(recombined);

figure;
imshow(RGB, []);
title('Original');

figure;
imshow(recombined_RGB);
title('Recombined');

%% Luma sub-sampling
yImage_reduced = imresize(Y, 1/2);
yImage_upsamp = imresize(yImage_reduced, 2);

recombined_2 = cat(3, yImage_upsamp, Cb, Cr);
figure;
imshow(recombined_2, []);
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

%% 4 Image Transform
T = dctmtx(8);
figure
imshow(T); title('DCT transform matrix');
lena = imread('lena.tiff');
f = double(rgb2gray(lena));

DCT = floor(blkproc(f-128,[8 8],'P1*x*P2',T,T'));
figure
imshow(abs(DCT),[]); title('F-trans');
figure
imshow(abs(DCT(81:88,297:304)), []); title('DCT sub-img at (81,297)');
figure
imshow(abs(DCT(1:8,1:8)), []); title('DCT sub-img at (1,1)');

mask = zeros(8,8);
mask ( 1 , 1 ) = 1 ;
mask ( 1 , 2 ) = 1 ;
mask ( 1 , 3 ) = 1 ;
mask ( 2 , 1 ) = 1 ;
mask ( 3 , 1 ) = 1 ;
mask ( 2 , 2 ) = 1 ;
F_thresh = blkproc(DCT, [8 8], 'P1.*x', mask);
f_thresh = floor(blkproc(F_thresh, [8 8], 'P1*x*P2',T',T))+128;

figure
imshow(f_thresh, []); title('reconstruct 3');
reconstruct_psnr = psnr(f_thresh,f);

figure
imshow(f,[]); title('original');

%% 5 Quatization
lena = imread('lena.tiff');
f = double(rgb2gray(lena));

DCT = floor(blkproc(f-128,[8 8],'P1*x*P2',T,T'));

Z = [16 11 10 16 24 40 51 61;
    12 12 14 19 26 58 60 55;
    14 13 16 24 40 57 69 56;
    14 17 22 29 51 87 80 62;
    18 22 37 56 68 109 103 77;
    24 35 55 64 81 104 113 92;
    49 64 78 87 103 121 120 101;
    72 92 95 98 112 100 103 99];

P1 = [1,3,5,10];
for i = 1:size(P1,2)
    quantized_DCT = round(blkproc(DCT, [8 8], 'P1/x', P1(i)*Z));
    unquantized_DCT = round(blkproc(quantized_DCT, [8 8], 'P1*x', P1(i)*Z));
    reconstruct_Z = floor(blkproc(unquantized_DCT, [8 8], 'P1*x*P2',T',T))+128;

    figure
    imshow(reconstruct_Z,[]); title(sprintf('reconstruct %dZ', P1(i)));
    reconstruct_psnr = psnr(reconstruct_Z,f);
end
