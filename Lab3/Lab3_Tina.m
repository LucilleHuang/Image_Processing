clc;
clear all;
close all;
%% Fourier Analysis
f=zeros(256, 256);
f(:,108:148)=1;

figure(1);
imshow(f, []); title('Test Image');

f_tf = fft2(f);
f_tf = abs(fftshift(f_tf));
figure(2);
imshow(f_tf, []); title('Fourier Spectra of Test Image');

f_r = imrotate(f, 45);
figure(3);
imshow(f_r, []); title('Test Image Rotated 45 Degree');

r_tf = abs(fftshift(fft2(f_r)));
figure(4)
imshow(r_tf, []); title('Fourier Spectra of Rotated Test Image')


lena = imread('lena.tiff');
figure(5)
imshow(lena); title('Original Lena Image');
lena = rgb2gray(lena);
lena_tf = fftshift(fft2(lena));
lena_amp = abs(lena_tf);
lena_phase = lena_tf./lena_amp;

i_lena_amp = log(abs(ifft2(ifftshift(lena_amp))));
i_lena_phase = ifft2(ifftshift(lena_phase));

figure(6)
imshow(i_lena_amp, []); title('Reconstructed Lena with Amplitude');

figure(7)
imshow(i_lena_phase, []); title('Reconstructed Lena with Phase');

%% Noise Reduction in the Frequency Domain
lena_intensity = im2double(lena);
lena_noisy = imnoise(lena_intensity,'gaussian',0,0.005);

figure(8)
imshow(lena_noisy); title('Noisy Lena');

lena_noisy_log = log(abs(fftshift(fft2(lena_noisy))));
figure(9)
imshow(lena_noisy_log, []); title('Log Fourier Spectra of Lena with Gaussian Noise');

lena_tf_log = log(lena_amp);
figure(10)
imshow(lena_tf_log,[]); title('Log Fourier Spectra of Original Lena');

lena_noisy_tf=fftshift(fft2(lena_noisy));
% Create an image of a white circle with radius r=60
r = 60;
h = fspecial('disk', r); h(h>0)=1;
% Create a black image
[height, width] = size(lena);
h_freq = zeros(height, width);
% Center the circle onto the black image
h_freq(height/2-r:height/2+r,width/2-r:width/2+r)=h;
% Create and plot Fourier Spectra of the low-pass filter
h_freq_tf=abs(fftshift(fft2(h_freq)));
figure(11)
imshow(h_freq_tf, []); title('Fourier Spectra of Low-Pass Filter, r=60');

filtered_noisy_lena = lena_noisy_tf.*h_freq;
filtered_noisy_lena_ift = abs(ifft2(ifftshift(filtered_noisy_lena)));
figure(12)
imshow(filtered_noisy_lena_ift, []); title('Denoised Lena with Low-Pass, r=60');
% PSNR
psnr_60 = psnr(filtered_noisy_lena_ift,lena_intensity)

% Create an image of a white circle with radius r=20
r = 20;
h = fspecial('disk', r); h(h>0)=1;
% Create a black image
[height, width] = size(lena);
h_freq = zeros(height, width);
% Center the circle onto the black image
h_freq(height/2-r:height/2+r,width/2-r:width/2+r)=h;
% Create and plot Fourier Spectra of the low-pass filter
h_freq_tf=abs(fftshift(fft2(h_freq)));
figure(13)
imshow(h_freq_tf, []); title('Fourier Spectra of Low-Pass Filter, r=20');

filtered_noisy_lena = lena_noisy_tf.*h_freq;
filtered_noisy_lena_ift = abs(ifft2(ifftshift(filtered_noisy_lena)));
figure(14)
imshow(filtered_noisy_lena_ift, []); title('Denoised Lena with Low-Pass, r=20');
% PSNR
psnr_20 = psnr(filtered_noisy_lena_ift,lena_intensity)

% Gaussian low-pass filter kernel with sigma=60
g_filter = fspecial('gaussian', height, 60);
g_filter_max = max(g_filter, [], 'all');
g_filter_norm = g_filter ./ g_filter_max;
figure(15)
imshow(g_filter_norm); title('Gaussian Low-Pass Filter Kernel, sigma=60');

% Perform Fourier Transform on Gaussian Filter
g_ft = fft2(g_filter_norm);
g_ft_shift = fftshift(abs(g_ft));
figure(16)
imshow(g_ft_shift, []);
title('Fourier Spectra of Gaussian');

% Question: Do we need to perform Fourier Transform on the filter?
g_filtered_noisy = lena_noisy_tf.*g_filter_norm;
g_filtered_noisy_ift = abs(ifft2(ifftshift(g_filtered_noisy)));
figure(17)
imshow(g_filtered_noisy_ift, []); title('Denoised Lena with Gaussian Low-Pass Filter');
%PSNR
psnr_G = psnr(g_filtered_noisy_ift,lena_intensity)

%% Filter Design
frequnoisy = imread('frequnoisy.tif');
figure;
imshow(frequnoisy); title('Original Frequnoisy.tif');
[M,N]=size(frequnoisy);
F=fft2(im2double(frequnoisy)); % taking the fast fourier transform of the image
F=fftshift(F);
log_F = log(abs(F));
figure(18)
imshow(log_F, []); title('Fourier Spectra of Frequnoisy Image');
NF = ones(size(F,1),size(F,2));

NF(65,65)=0;
NF(193,193)=0;
NF(139,153)=0;
NF(119,105)=0;
figure(19)
imshow(NF); title('Notch Filter');
G = F.*NF;
G=ifftshift(G);
g=real(ifft2(double(G)));
figure(20)
imshow(g,[ ]); title('Denoised Image');

