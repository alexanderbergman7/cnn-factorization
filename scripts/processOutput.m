output = load('../outputIm.mat');
outputF = load('../filenames.mat');
images = output.outputIm;
filenames = outputF.filenames;
validation = dir('../../dataset/images/val_gray/*.jpg');

for i = 1:size(images,1)
   assert(all(strcat(filenames(i,:)) == validation(i).name))
   filename = validation(i).name;
   
   imageR = im2double(uint8(squeeze(images(i,:,:))));
   image = im2double(imread(strcat('../../dataset/images/val_gray/',filename)));
   
   MSE = (1/(size(image,1)*size(image,2)))*sum(sum((image-imageR).^2));
   PSNR = 10*log10((max(image(:))^2)/MSE);
   
   val_results(i).name = filename;
   val_results(i).PSNR = PSNR;
   val_results(i).MSE = MSE;
end

save('../val_results_E25_BS40_ic1_oc1_f1_d2.mat', 'val_results')