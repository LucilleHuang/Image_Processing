clc;
clear all;
close all;

lena_original = imread('lena.tiff');
lena = rgb2gray(lena_original);
camman = imread('cameraman.tif');

%% Digital Zooming

% reduce resolution by factor of 4
lena_g_shrink = imresize(lena,1/4,'bilinear');
camman_shrink = imresize(camman,1/4,'bilinear');

figure; imshow(lena_g_shrink); title('Lena shrink')
figure; imshow(camman_shrink); title('cameraman shrink')

% nearest neighbor
lena_nn = imresize(lena_g_shrink, 4, 'nearest');
camman_nn = imresize(camman_shrink, 4, 'nearest');

figure; imshow(lena_nn); title('Lena nearest')
figure; imshow(camman_nn); title('cameraman nearest')

% bilinear interpolation
lena_bi = imresize(lena_g_shrink, 4, 'bilinear');
camman_bi = imresize(camman_shrink, 4, 'bilinear');

figure; imshow(lena_bi); title('Lena bilinear')
figure; imshow(camman_bi); title('cameraman bilinear')

% bicubical interpolation
lena_cub = imresize(lena_g_shrink, 4, 'bicubic');
camman_cub = imresize(camman_shrink, 4, 'bicubic');

figure; imshow(lena_cub); title('Lena bicubical')
figure; imshow(camman_cub); title('cameraman bicubical')

%PSNR values
lena_nn_psnr = psnr(lena, lena_nn)
lena_bi_psnr = psnr(lena, lena_bi)
lena_cub_psnr = psnr(lena, lena_cub)

camman_nn_psnr = psnr(camman, camman_nn)
camman_bi_psnr = psnr(camman, camman_bi)
camman_cub_psnr = psnr(camman, camman_cub)

%% 4 Point Operations
tire = imread('tire.tif');
figure;
subplot(2,1,1); 
imshow(tire)
subplot(2,1,2); 
imhist(tire)
ylim([0 2100]);
xlim([-4 255]);

% Negative Trans.
tire_neg = (255) - tire;
figure;
subplot(2,1,1); 
imshow(tire_neg)
subplot(2,1,2); 
imhist(tire_neg)
ylim([0 2100]);
xlim([-4 255]);

% Power-law Trans
tire_p1 = imadjust(tire,[],[],0.5);
figure;
subplot(2,1,1); 
imshow(tire_p1)
subplot(2,1,2); 
imhist(tire_p1)
ylim([0 2100]);
xlim([-4 255]);

tire_p2 = imadjust(tire,[],[],1.3);
figure;
subplot(2,1,1); 
imshow(tire_p2)
subplot(2,1,2); 
imhist(tire_p2)
ylim([0 4500]);
xlim([-4 255]);

% histogram equalization
tire_eq = histeq(tire);
figure;
subplot(2,1,1); 
imshow(tire_eq)
subplot(2,1,2); 
imhist(tire_eq)
ylim([0 2100]);
xlim([-4 255]);