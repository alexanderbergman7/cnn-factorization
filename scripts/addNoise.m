validation = dir('val/*.jpg');
train = dir('train/*.jpg');

nvals = length(validation);
ntrain = length(train);

for i=1:nvals
   filename = validation(i).name;
   image = im2double(rgb2gray(imread(strcat('val/',filename))));
   % add noise
   image = image + .1*randn(size(image));
   image = image(1:end-1, 1:end-1);
   
   imwrite(image, strcat('noisy_val_sigma01/', filename));
end

for i=1:ntrain
   filename = train(i).name;
   image = im2double(rgb2gray(imread(strcat('train/',filename))));
   % add noise
   image = image + .1*randn(size(image));
   image = image(1:end-1, 1:end-1);
   
   imwrite(image, strcat('noisy_train_sigma01/', filename));
end