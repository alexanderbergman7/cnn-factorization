% in order for this script to run, must download the BM3D codebase at 
% http://www.cs.tut.fi/~foi/GCF-BM3D/ and add the BM3D function to the
% filepath
validation = dir('../../dataset/images/val_gray/*.jpg');
validationNoisy = dir('../../dataset/images/noisy_val_sigma01/*.jpg');

nvals = length(validation);

% loop through validation files/testing files to evaluate with BM3D
for i=1:nvals
   filename = validation(i).name;
   image = im2double(imread(strcat('../../dataset/images/val_gray/',filename)));
   image_noisy = im2double(imread(strcat('../../dataset/images/noisy_val_sigma01/',filename)));
   
   % calculate PSNR of BM3D denoised image
   [PSNR, est] = BM3D(image, image_noisy, .1*255, 'np', 0);
   MSE = (1/(size(est,1)*size(est,2)))*sum(sum((image-est).^2));
   
   % store in results
   val_results(i).name = filename;
   val_results(i).PSNR = PSNR;
   val_results(i).MSE = MSE;
   
   imwrite(est, strcat('../../dataset/images/noisy_val_sigma01_BM3D_denoised/', filename));
end

save('../../dataset/images/noisy_val_sigma01_BM3D_denoised/val_results.mat', 'val_results')