function trainAndTestVGG16(datasetPath)

% Image Category Classification Using Deep Learning

% % Store the output in a temporary folder
% datasetPath = '..\Dataset';

% Uncompressed data set
imageFolder = fullfile(datasetPath);

% Load Images
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', ...
    'IncludeSubfolders',true, 'FileExtensions','.png');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

% Load pretrained network
net = vgg16();
% net = vgg19();

save dataNet net;

% Other popular networks trained on ImageNet include AlexNet, GoogLeNet,
% VGG-16 and VGG-19 [3], which can be loaded using alexnet, googlenet,
% vgg16, and vgg19 from the Deep Learning Toolbox™.

% Visualize the first section of the network.
% figure
% plot(net)
% title('First section of Vgg-16')
% set(gca,'YLim',[150 170]);

analyzeNetwork(net)

% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)
% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames)

[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

% Create augmentedImageDatastore from training and test sets to resize
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestAll = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');

% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')
 
featureLayer = 'fc8';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Evaluate Classifier
% Repeat the procedure used earlier to extract image features from testSet. The test features can then be passed to the classifier to measure the accuracy of the trained classifier.

% Extract test features using the CNN
testAllFeatures = activations(net, augmentedTestAll, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
  
%  Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testAllFeatures, 'ObservationsIn', 'columns');
 
% Tabulate the results using a confusion matrix.
[confMat, order] = confusionmat(trueLabels, predictedLabels);
        save ('dataClassifier', 'classifier', 'predictedScoresNew');

% Display the mean accuracy
disp('mean accuracy')
mean(diag(confMat)*100)

figure
[confMat] = confusionchart(trueLabels,predictedLabels, ...
    'Title','Confussion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

