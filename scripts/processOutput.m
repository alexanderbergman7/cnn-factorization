output = load('../outputIm.mat');
outputF = load('../filenames.mat');
images = output.outputIm;
filenames = outputF.filenames;

for i = 1:size(images,1)
   filename = strcat(filenames(i,:));
   
   imageR = im2double(uint8(squeeze(images(i,:,:))));
   image = im2double(imread(strcat('../../dataset/images/val_gray/',filename)));
   
   MSE = (1/(size(image,1)*size(image,2)))*sum(sum((image-imageR).^2));
   PSNR = 10*log10((max(image(:))^2)/MSE);
   SSIM = ssim(imageR, image);
   
   val_results(i).name = filename;
   val_results(i).PSNR = PSNR;
   val_results(i).MSE = MSE;
   val_results(i).SSIM = SSIM;
end

save('../test_model_env/val_results.mat', 'val_results')