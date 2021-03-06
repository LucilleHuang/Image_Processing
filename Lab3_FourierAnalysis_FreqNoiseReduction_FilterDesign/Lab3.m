clc;
clear all;
close all;

%% Fourier Analysis
test_img=zeros(256,256);
test_img(:,108:148)=1;
% figure
% imshow(test_img); title('Test Image');

test_fourier = fft2(test_img);%fourier transform
test_fourier = fftshift(test_fourier);%shift origin
test_fourier = abs(test_fourier);%amplitude
figure
imshow(test_fourier,[]); title('Fourier Spectra of Test Image');

% rotate 45 degree
test_img_rotate = imrotate(test_img,45);
figure
imshow(test_img_rotate); title('Test Image Rotated 45 Degree');

test_rot_fourier = fft2(test_img_rotate);%fourier transform
test_rot_fourier = fftshift(test_rot_fourier);%shift origin
test_rot_fourier = abs(test_rot_fourier);%amplitude
figure
imshow(test_rot_fourier,[]); title('Fourier Spectra of Rotated Test Image');

%% amplitude and phase
lena = imread('lena.tiff');
lena_grey = rgb2gray(lena);
lena_fourier = fft2(lena_grey);%fourier transform
lena_fourier = fftshift(lena_fourier);%shift origin
lena_amp = abs(lena_fourier);%amplitude
lena_phase = lena_fourier./lena_amp;
lena_amp_inverse = ifft2(lena_amp);
lena_phase_inverse = ifft2(lena_phase);

% figure
% imshow(lena_grey);title('Original Lena Image');
% figure
% imshow(lena_amp_inverse, []);title('Reconstructed Lena with Amplitude');
% figure
% imshow(lena_phase_inverse, []);title('Reconstructed Lena with Phase');

%% Noise Reduction in Freq. Domain
lena_in = double(lena_grey)./255;
lena_in_fourier = fft2(lena_in);
lena_in_fourier = fftshift(lena_in_fourier);
lena_in_fourier_log = log(lena_in_fourier);

lena_gaus = imnoise(lena_in, 'gaussian', 0, 0.005);
lena_gaus_fourier = fft2(lena_gaus);
lena_gaus_fourier = fftshift(lena_gaus_fourier);
lena_gaus_fourier_log = log(lena_gaus_fourier);

% figure
% imshow(lena_in_fourier_log);title('Log Fourier Spectra of Original Lena');
% figure
% imshow(lena_gaus_fourier_log, []);title('Log Fourier Spectra of Lena with Gaussian Noise');

% % low-pass filter: Why are PSNR complex?
info = imfinfo('lena.tiff');
height = info.Height;
width = info.Width;

r=60;
h=fspecial('disk',r); h(h>0)=1;
lowpass_r60 = zeros([height],[width]);
lowpass_r60([[height]/2-r:[height]/2+r],[[width]/2-r:[width]/2+r])=h;
lena_gaus_fourier_lowpass_r60 = imfilter(lena_gaus_fourier,lowpass_r60);
lena_gaus_inverse_1 = ifft2(lena_gaus_fourier_lowpass_r60);
figure
imshow(lena_gaus_inverse_1);title('Denoised Lena with Lowpass r=60');
lena_gaus_inverse_psnr_1 = psnr(lena_gaus_inverse_1,lena_in)

r=20;
h=fspecial('disk',r); h(h>0)=1;
lowpass_r20 = zeros([height],[width]);
lowpass_r20([[height]/2-r:[height]/2+r],[[width]/2-r:[width]/2+r])=h;
lena_gaus_fourier_lowpass_r20 = imfilter(lena_gaus_fourier,lowpass_r20);
lena_gaus_inverse_2 = ifft2(lena_gaus_fourier_lowpass_r20);

figure
imshow(lena_gaus_inverse_2);title('Denoised Lena with Lowpass r=20');
lena_gaus_inverse_psnr_2 = psnr(lena_gaus_inverse_2, lena_in)

%Gaussian Filter
Gau_filter= fspecial('gaussian', height , 60);
max_val = max(Gau_filter, [], 'all');
Gau_filter_normalzied = Gau_filter./max_val;
lena_filtered_by_gau = imfilter(lena_gaus_fourier,Gau_filter_normalzied);
lena_gaus_inverse_3 = ifft2(lena_filtered_by_gau);
figure
imshow(lena_gaus_inverse_3);title('Denoised Lena with Gaussian Filter');
lena_gaus_inverse_psnr_3 = psnr(lena_gaus_inverse_3, lena_in)

%% Filter Design
frequnoisy = imread('frequnoisy.tif');
figure;
imshow(frequnoisy); title('Original Frequnoisy.tif');
[M,N]=size(frequnoisy);
F=fft2(double(frequnoisy)); % taking the fast fourier transform of the image
F=fftshift(F);
log_F = log(F);
figure;
imshow(log_F, []); title('Fourier Spectra of Frequnoisy Image');
notch_filter = ones(size(F,1),size(F,2));

notch_filter(65,65)=0;
notch_filter(193,193)=0;
notch_filter(139,153)=0;
notch_filter(119,105)=0;
figure;
imshow(notch_filter); title('Notch Filter');
G = F.*notch_filter;
G=ifftshift(G);
g=real(ifft2(double(G)));
figure;
imshow(g,[ ]); title('Denoised Image');