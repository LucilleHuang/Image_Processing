clc;
clear all;
close all;

lena = imread('lena.tiff');
camman = imread('cameraman.tif');
figure
imshow(camman); title('Original cameraman.tif');

% Discrete Convolution for Image Processing
lena_grey = rgb2gray(lena);
figure
imshow(lena_grey); title('Figure 1: Grayscale Lena.tiff');

% intensity of the image
lena_in = double(lena_grey)./255;
camman_in = double(camman)./255;

% three impulse functions
h1 = (1/6)*ones(1,6);
h2 = h1';
h3 = [-1 1];

% lena image with convolution
lena_conv_1 = conv2(lena_in, h1);
figure
imshow(lena_conv_1); title('Figure 2: Lena.tiff convolve with h1 impulse function');

lena_conv_2 = conv2(lena_in, h2);
figure
imshow(lena_conv_2); title('Figure 3: Lena.tiff convolve with h2 impulse function');

lena_conv_3 = conv2(lena_in, h3);
figure
imshow(lena_conv_3); title('Figure 4: Lena.tiff convolve with h3 impulse function');

%% Noise Generation
% toy image
f = [0.3*ones(200, 100) 0.7*ones(200,100)];
figure
imshow(f); title('Original toy image');
figure
imhist(f);
title('original toy image histogram');

% additive zero-mean Gaussian with variance of 0.01
toy_zero_mean = imnoise(f, 'gaussian');
figure
imshow(toy_zero_mean);
title('Figure 5: toy with additive zero-mean Gaussian(variance = 0.01)');
figure
imhist(toy_zero_mean); 
title('Figure 6: additive zero-mean Gaussian histogram');

% salt and pepper with noise density of 0.05
toy_salt_and_pepper = imnoise(f, 'salt & pepper', 0.05);
figure
imshow(toy_salt_and_pepper);
title('Figure 7: toy with salt and pepper (density = 0.05)');
figure
imhist(toy_salt_and_pepper); xlim([-0.03 1.03])
title('Figure 8: salt and pepper histogram');

% multiplicative speckle noise (variance = 0.04)
toy_mul_spec = imnoise(f, 'speckle', 0.04);
figure
imshow(toy_mul_spec);
title('Figure 9: toy with multiplicative speckle noise');
figure
imhist(toy_mul_spec);
title('Figure 10: multiplicative speckle histogram');

%% Noise Reduction in the Spatial Domain
%lena image with zeor-mean Gaussian noise with variance of 0.002
lena_gaus = imnoise(lena_in, 'gaussian', 0, 0.002);
figure
imshow(lena_gaus); title('Figure 11: Lena.tiff with zero-mean Gaussian noise (variance = 0.002)');
figure
imhist(lena_gaus); title('Figure 12: Lena.tiff with zero-mean Gaussian (variance = 0.002)');
lena_gaus_psnr = psnr(lena_in, lena_gaus)

% average filtering 3x3
average_filter_3 = fspecial('average', 3);
figure
imagesc(average_filter_3); title('Figure 13: 3x3 averaging filter');
colormap(gray); 

lena_denoised_ave_3 = imfilter(lena_gaus, average_filter_3);
figure
imshow(lena_denoised_ave_3); title('Figure 14: denoised Lena with average filter 3x3');
figure
imhist(lena_denoised_ave_3); title('Figure 15: histogram for Lena denoiced with averaging filter 3x3');
lena_denoised_ave_3_psnr = psnr(lena_in, lena_denoised_ave_3)

imhist(lena_in); title('Figure 16: histogram of original lena.tiff');

% average filtering 7x7
average_filter_7 = fspecial('average', 7);

lena_denoised_ave_7 = imfilter(lena_gaus, average_filter_7);
figure
imshow(lena_denoised_ave_7); title('Figure 17: denoised Lena with average filter 7x7');
figure
imhist(lena_denoised_ave_7); title('Figure 18: histogram for Lena denoiced with averaging filter 7x7');
lena_denoised_ave_7_psnr = psnr(lena_in, lena_denoised_ave_7)


% Guassian 7x7
Gau_filter_7 = fspecial('gaussian', 7);
figure
imagesc(Gau_filter_7); title('Figure 19: 7x7 Gaussian filter');
colormap(gray); 

lena_denoised_Gau_7 = imfilter(lena_gaus, Gau_filter_7);
figure
imshow(lena_denoised_Gau_7); title('Figure 20: denoised Lena with Gaussian filter 7x7');
figure
imhist(lena_denoised_Gau_7); title('Figure 21: histogram for Lena denoiced with Gaussian filter 7x7');
lena_denoised_Gau_7_psnr = psnr(lena_in, lena_denoised_Gau_7)

% Lena with salt and pepper with noise density of 0.05
lena_salt_and_pepper = imnoise(lena_in, 'salt & pepper', 0.05);
figure
imshow(lena_salt_and_pepper);
title('Figure 22: Lena with salt and pepper (density = 0.05)');
figure
imhist(lena_salt_and_pepper); ylim([0 7000]); xlim([-0.03 1.03])
title('Figure 23: Lena with salt and pepper histogram');

% average filtering 7x7
average_filter_7 = fspecial('average', 7);

lena_salt_denoised_ave_7 = imfilter(lena_salt_and_pepper, average_filter_7);
figure
imshow(lena_salt_denoised_ave_7); title('Figure 24: denoised Lena with average filter 7x7');
figure
imhist(lena_salt_denoised_ave_7); title('Figure 25: histogram for Lena denoiced with averaging filter 7x7');
lena_denoised_ave_7_psnr = psnr(lena_in, lena_salt_denoised_ave_7)


% Guassian 7x7
Gau_filter_7 = fspecial('gaussian', 7);

lena_salt_denoised_Gau_7 = imfilter(lena_salt_and_pepper, Gau_filter_7);
figure
imshow(lena_salt_denoised_Gau_7); title('Figure 26: denoised Lena with Gaussian filter 7x7');
figure
imhist(lena_salt_denoised_Gau_7); title('Figure 27: histogram for Lena denoiced with Gaussian filter 7x7');
lena_denoised_Gau_7_psnr = psnr(lena_in, lena_salt_denoised_Gau_7)

% median filter
lena_salt_denoised_med = medfilt2(lena_salt_and_pepper);
figure
imshow(lena_salt_denoised_med); title('Figure 28: denoised Lena with median filter');
figure
imhist(lena_salt_denoised_med); title('Figure 29: histogram for Lena denoiced with median filter');
lena_denoised_med_psnr = psnr(lena_in, lena_salt_denoised_med)

% Sharpening in the Spatial Domain
Gau_filter_7 = fspecial('gaussian', 7);

camman_Gau_7 = imfilter(camman_in, Gau_filter_7);
figure
imshow(camman_Gau_7); title('Figure 30: Cameraman with Gaussian filter 7x7');

camman_sub = camman_in - camman_Gau_7;
figure
imshow(camman_sub); title('Figure 31: Cameraman subtracting Gaussian filter 7x7');

camman_original_plus_sub = camman_in + camman_sub;
figure
imshow(camman_original_plus_sub); title('Figure 32: Cameraman original plus subtracted');

camman_original_plus_half_sub = camman_in + camman_sub*0.5;
figure
imshow(camman_original_plus_half_sub); title('Figure 33: Cameraman original plus subtracted*0.5');

