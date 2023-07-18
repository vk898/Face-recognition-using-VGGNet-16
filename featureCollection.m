function featureCollection(datasetPath)

% Store the output in a temporary folder
% datasetPath = '..\Dataset';

% Uncompressed data set
imageFolder = fullfile(datasetPath);

% Load Images
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true, 'FileExtensions','.jpg');

hWaitBar = waitbar(0, 'Please wait...', 'Name','Creating Database.', 'Position', [500 80 280 50]);

numFiles = length(imds.Files);
for i = 1:numFiles
    close all;
    
    filePath = imds.Files{i};
    
    imgData = imread(filePath);
    imgData = imresize(imgData, [240, 320]);
    figure, imshow(imgData), title('Input Image');
    
    [imgFace, bbox] = faceSegmentation(imgData);
    if (isempty(bbox) || (size(bbox,1) > 1))
        continue;
    end
    figure, imshow(imgFace), title('Detected Face');
    
    croppedFace = imcrop(imgData, bbox);
    croppedFace = imresize(croppedFace, [200, 200]);
    figure, imshow(croppedFace), title('Cropped Face');
    
    [imgPath, imgName, imgExt] = fileparts(filePath);
    
    filePath = fullfile(imgPath, [imgName,'.png']);
    imwrite(croppedFace, filePath);
    
    waitbar(i/numFiles, hWaitBar);
    pause(0.1);
end

close(hWaitBar);

disp('Done: Face Segmentation ...');

return;
