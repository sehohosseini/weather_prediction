clc
clear
close all
format compact
%% import data
data = readtable("seattle-weather.csv");
trainingData = data(1:1022,:);
X_test = data(1023:end,2:5);
Y_test = data(1023:end,6);
%% preparing data
inputTable = trainingData;
predictorNames = {'precipitation', 'temp_max', 'temp_min', 'wind'};
predictors = inputTable(:, predictorNames);
response = inputTable.weather;
isCategoricalPredictor = [false, false, false, false];
classNames = {'drizzle'; 'fog'; 'rain'; 'snow'; 'sun'};

%% training the model

classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'on', ...
    'ClassNames', classNames);

predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));
trainedClassifier.RequiredVariables = {'precipitation', 'temp_max', 'temp_min', 'wind'};
trainedClassifier.ClassificationTree = classificationTree;

inputTable = trainingData;
predictorNames = {'precipitation', 'temp_max', 'temp_min', 'wind'};
predictors = inputTable(:, predictorNames);
response = inputTable.weather;
isCategoricalPredictor = [false, false, false, false];
classNames = {'drizzle'; 'fog'; 'rain'; 'snow'; 'sun'};
%% testing the model

classificationTree = fitctree(...
    X_test, ...
    Y_test, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'on', ...
    'ClassNames', classNames);

% createing the result struct with predict function
treePredictFcn = @(x) predict(classificationTree, x);
testPredictFcn = @(x) treePredictFcn(x);
%% evaluating the model

[validationPredictions, validationScores] = testPredictFcn(X_test);

% Compute test accuracy
X_test = table2array(X_test);
Y_test = table2array(Y_test);
correctPredictions = strcmp( strtrim(validationPredictions), strtrim(Y_test));
isMissing = cellfun(@(x) all(isspace(x)), Y_test, 'UniformOutput', true);
correctPredictions = correctPredictions(~isMissing);
testAccuracy = sum(correctPredictions)/length(correctPredictions);

disp(['The accuracy of the model is equal to = ' num2str(testAccuracy)])