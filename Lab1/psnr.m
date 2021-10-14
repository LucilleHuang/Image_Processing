function psnr_out = psnr(f,g)
[m n] = size(f); % get rows and cols of image_f
Max_f = 255.0;
MSE = 1/(m*n)*sum(sum((double(f)-double(g)).^2));
psnr_out = 10*log10(Max_f^2/MSE);