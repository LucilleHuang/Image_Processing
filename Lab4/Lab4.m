clc;
clear all;
close all;

%% Image Restoration in the Frequency Domain

% disk blur with r=4
h_d = fspecial('disk',4);
h = zeros(256,256);
h(1:9,1:9) = h_d;
h = circshift(h, [-5,-5]);
f = im2double(imread('cameraman.tif'));
imshow(f); title('original image');

h_freq = fft2(h);
f_blur = real(ifft2(h_freq.*fft2(f)));

figure
imshow(f_blur); title('blurred image');
f_blur_psnr = psnr(f_blur, f)

inversed_f_blur = fft2(f_blur)./h_freq;
inversed_f_blur = real(ifft2(inversed_f_blur));

figure
imshow(inversed_f_blur); title('inversed blurred image');
inversed_psnr = psnr(abs(inversed_f_blur), f)

% guassian noise with 0.002 variance
f_gau = imnoise(f_blur, 'gaussian', 0.002);

inversed_f_gau = fft2(f_gau)./h_freq;
inversed_f_gau = real(ifft2(inversed_f_gau));

figure
imshow(inversed_f_gau); title('inversed blurred image with gaussian noise');
inversed_f_gau_psnr = psnr(abs(inversed_f_gau), f)

% Wiener filter
estimated_nsr = 0.002 / var(f_gau(:));
f_wiener = deconvwnr(f_gau, fftshift(h), estimated_nsr);

figure
imshow(f_wiener); title('Wiener filtered blurred image with gaussian noise');
f_wiener_psnr = psnr(abs(f_wiener), f)

%% Adaptive Filtering
f = im2double(imread('cameraman.tif'));
degraded = im2double(imread('degraded.tif'));
figure
imshow(degraded); title('degraded image');

local_mean = colfilt(degraded, [5 5], 'sliding', @mean);
local_var = colfilt(degraded, [5 5], 'sliding', @var);
% flat_region values come from reading the X and Y coordinates from Data Tip of the top right corner of the original image
flat_region = degraded(1:100, 180:256);
noise_var = var(flat_region(:));

K = (local_var - noise_var)./local_var;
f_est = K.*degraded+(1-K).*local_mean;
figure
imshow(f); title('original image');
figure
imshow(f_est); title('denoised image using Lee filter with noise variance = 0.0109');
filtered_by_Lee_psnr = psnr(f_est, f)

%Gaussian lowpass Filter
info = imfinfo('degraded.tif');
height = info.Height;
g_filter = fspecial('gaussian', height, 30);
g_filter_max = max(g_filter, [], 'all');
g_filter_norm = g_filter ./ g_filter_max;

degraded_fourier = fftshift(fft2(degraded));
filtered_by_gau = degraded_fourier.*g_filter_norm;
filtered_by_gau = abs(ifft2(ifftshift(filtered_by_gau)));
figure
imshow(filtered_by_gau); title('denoised image using Gaussian lowpass filter');
filtered_by_gau_psnr = psnr(filtered_by_gau, f)

% higher and lower noise variance
noise_var_higher = 0.0209;
K2 = (local_var - noise_var_higher)./local_var;
f_est2 = K2.*degraded+(1-K2).*local_mean;
figure
imshow(f_est2); title('denoised image using Lee filter with noise variance = 0.0209');
filtered_by_Lee_higher_var_psnr = psnr(f_est2, f)

noise_var_lower = 0.0009;
K3 = (local_var - noise_var_lower)./local_var;
f_est3 = K3.*degraded+(1-K3).*local_mean;
figure
imshow(f_est3); title('denoised image using Lee filter with noise variance = 0.0009');
filtered_by_Lee_lower_var_psnr = psnr(f_est3, f)

% smaller window
% The 3x3 window gives psnr = NaN due to a "NaN/Inf breakpoint hit for psnr.m on line 75" error. 
% Therefore, a 4x4 window was used
local_mean = colfilt(degraded, [4 4], 'sliding', @mean);
local_var = colfilt(degraded, [4 4], 'sliding', @var);

K = (local_var - noise_var)./local_var;
f_est4 = K.*degraded+(1-K).*local_mean;
filtered_by_Lee_psnr_4x4 = psnr(f_est4, f)
figure
imshow(f_est4); title('denoised with Lee filter using 4x4 window');

% bigger window
local_mean = colfilt(degraded, [7 7], 'sliding', @mean);
local_var = colfilt(degraded, [7 7], 'sliding', @var);

K = (local_var - noise_var)./local_var;
f_est5 = K.*degraded+(1-K).*local_mean;
figure
imshow(f_est5); title('denoised with Lee filter using 7x7 window');
filtered_by_Lee_psnr_7x7 = psnr(f_est5, f)

