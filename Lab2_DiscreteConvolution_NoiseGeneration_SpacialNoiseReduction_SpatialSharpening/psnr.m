function psnr_out = psnr(f, g)
psnr_out = 10*log10(1/mean2((f-g).^2));
end