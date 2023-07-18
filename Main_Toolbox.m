close all;
clear all;
clc;

Repeat = 1;
while(Repeat)
    choice = menu('Person Identification - Face', ...
        'Face Segmentation', 'VGG16 Bulk Testing', 'Select Query', ...
        'Exit');
    
    switch choice
        
        case 1
            disp('Feature Extraction of Database');

            % Store the output in a temporary folder
            datasetPath = '..\Dataset\Training';
            
            featureCollection(datasetPath);
            
        case 2
            disp('VGG-16 Bulk Training and Testing');
            
            % Store the output in a temporary folder
            datasetPath = '..\Dataset\Training';
            
            trainAndTestVGG16(datasetPath);
            
        case 3
            disp('Select Query');
            
            % Store the output in a temporary folder
            datasetPath = '..\Dataset\Testing';
            
            selectQuery(datasetPath);
            
        case 4
            close all;
            clear all;
            clc;
            
            Repeat = 0;
    end
end

disp('Have a Nice Day...');

