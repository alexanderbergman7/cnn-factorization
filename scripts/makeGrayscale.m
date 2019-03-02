validation = dir('val/*.jpg');
train = dir('train/*.jpg');

nvals = length(validation);
ntrain = length(train);

for i=1:nvals
   filename = validation(i).name;
   image = im2double(rgb2gray(imread(strcat('val/',filename))));
   image = image(1:end-1, 1:end-1);
   imwrite(image, strcat('val_gray/', filename));
end

for i=1:ntrain
   filename = train(i).name;
   image = im2double(rgb2gray(imread(strcat('train/',filename))));
   image = image(1:end-1, 1:end-1);
   imwrite(image, strcat('train_gray/', filename));
end