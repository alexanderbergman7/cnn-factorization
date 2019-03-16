validation = dir('../../dataset/images/val/*.jpg');
train = dir('../../dataset/images/train/*.jpg');

nvals = length(validation);
ntrain = length(train);

for i=1:nvals
   filename = validation(i).name;
   image = im2double(rgb2gray(imread(strcat('../../dataset/images/val/',filename))));
   image = image(1:end-1, 1:end-1);
   
   % rotate image if it is tall
   if size(image) == [480 320]
       image = image';
   end
   
   % write grayscale
   imwrite(image, strcat('../../dataset/images/val_gray/', filename));
   
   % add noise
   imageN = image + .1*randn(size(image));
   
   imwrite(imageN, strcat('../../dataset/images/noisy_val_sigma01/', filename));
   
   MSE = (1/(size(image,1)*size(image,2)))*sum(sum((image-imageN).^2));
   PSNR = 10*log10((max(image(:))^2)/MSE);
   
   val_results(i).name = filename;
   val_results(i).PSNR = PSNR;
   val_results(i).MSE = MSE;
end

for i=1:ntrain
   filename = train(i).name;
   image = im2double(rgb2gray(imread(strcat('../../dataset/images/train/',filename))));
   image = image(1:end-1, 1:end-1);
   
   % rotate image if it is tall
   if size(image) == [480 320]
       image = image';
   end
   
   % write grayscale
   imwrite(image, strcat('../../dataset/images/train_gray/', filename));
   
   % add noise
   imageN = image + .1*randn(size(image));
   
   imwrite(imageN, strcat('../../dataset/images/noisy_train_sigma01/', filename));
   
   MSE = (1/(size(image,1)*size(image,2)))*sum(sum((image-imageN).^2));
   PSNR = 10*log10((max(image(:))^2)/MSE);
   
   train_results(i).name = filename;
   train_results(i).PSNR = PSNR;
   train_results(i).MSE = MSE;
end

save('../../dataset/images/noisy_train_sigma01/train_results.mat', 'train_results')
save('../../dataset/images/noisy_val_sigma01/val_results.mat', 'val_results')