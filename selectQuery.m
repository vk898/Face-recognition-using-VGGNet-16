function selectQuery(datasetPath)

close all;

%Read the image from dataset
[fileName, pathName] = uigetfile('*.jpg','Select Test Image', datasetPath);
if ~fileName
    errordlg('No file is selected');
    return;
end
fileName
filePath = [pathName fileName];

    imgData = imread(filePath);
    imgData = imresize(imgData, [240, 320]);
    figure, imshow(imgData), title('Input Image');
    
    [imgFace, bbox] = faceSegmentation(imgData);
    if (isempty(bbox) || (size(bbox,1) > 1))
        msgbox('Face NOT Detected');
        return;
%         continue;
    end
    figure, imshow(imgFace), title('Detected Face');

    croppedFace = imcrop(imgData, bbox);
    croppedFace = imresize(croppedFace, [200, 200]);
    figure, imshow(croppedFace), title('Cropped Face');

% [inImg]= imread(filePath);
% % inImg = imresize(inImg,[256 256]);
% figure, imshow(inImg), title('Input Image');

% [irisRegion, irisRubberSheetModel] = irisSegmentation(inImg);
% % figure, imshow(irisRegion);
% figure, imshow(irisRubberSheetModel);

load dataNet;
imageSize = net.Layers(1).InputSize;

featureLayer = 'fc8';
MiniBatchSize = 16;

load dataClassifier;

%% preprocessing
imgData = imresize(croppedFace, [imageSize(1), imageSize(2)]);
%     figure,imshow(inImg),title('Resized Image');

augmentedTestSetT = augmentedImageDatastore(imageSize, imgData, 'ColorPreprocessing', 'gray2rgb');

% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSetT, featureLayer, ...
    'MiniBatchSize', MiniBatchSize, 'OutputAs', 'columns');

%  Pass CNN image features to trained classifier
[predictedLabel, predictedScore] = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

predictedLabel = findPredictionLabel(predictedScoresNew, predictedScore);

if (predictedLabel > 0)
    predictedLabel = classifier.ClassNames(predictedLabel);
    resultString = sprintf('Predicted Person %s', predictedLabel);
else
    resultString = sprintf('Person NOT Identified');
end

msgbox(resultString);
