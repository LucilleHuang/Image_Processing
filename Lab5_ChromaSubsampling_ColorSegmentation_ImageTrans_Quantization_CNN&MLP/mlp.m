clc
clearvars


load mnist_.mat
Xtrain = double(mnist.train_images)/255.0 - 0.5;
Xtrain = reshape(Xtrain,28,28,1,5000);
ytrain = categorical(mnist.train_labels);
Xtest = double(mnist.test_images)/255.0 - 0.5;
Xtest = reshape(Xtest,28,28,1,10000);
ytest = categorical(mnist.test_labels);

for i = 1:25
    subplot(5,5,i);
    imshow(Xtrain(:,:,i),[]);
end

% Define Network Architecture
layers = [
    imageInputLayer([28 28 1])
    
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(10)
		   
    softmaxLayer
    classificationLayer];
cell_test = {Xtest ytest};
% Specify Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train Network Using Training Data
net = trainNetwork(Xtrain, ytrain, layers, options);

% Classify Validation Images and Compute Accuracy
YPred = classify(net,Xtest);
'Testing accuracy'
test_accuracy = sum(YPred == ytest)/numel(ytest)

'Training accuracy'
YPred = classify(net, Xtrain);
train_accuracy = sum(YPred == ytrain)/numel(ytrain)
