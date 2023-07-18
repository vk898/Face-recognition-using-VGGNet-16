function [imgFace, bbox] = faceSegmentation(imgData)

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a image and run the detector.
bbox            = step(faceDetector, imgData);

% Draw the returned bounding box around the detected face.
imgFace = insertObjectAnnotation(imgData,'rectangle',bbox,'Face');
% figure, imshow(imgFace), title('Detected face');
