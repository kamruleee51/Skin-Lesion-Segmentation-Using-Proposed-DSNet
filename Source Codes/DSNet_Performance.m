%% This code is dedicated for the processing of predicted mask 
%  from DSNet where ISODATA has been used for the mask thresholding. 
%  Finally, the biggest region has been selected as the lesion mask. 
%  For more details contact with the author at kamruleeekuet@gmail.com
%% Clean the environment 
clc;
clear all;
close all;
tic;

%% Load the images from the directory and initialize the performance metrics 
Path_Predicted_DSNet_Mask = [pwd,'/Mask/'];
Path_ISIC_2017_GT = [pwd,'/Test_GT_ISIC_2017/'];
Save_Directory=[pwd,'/Save/'];

Pred_Mask = dir(fullfile(Path_Predicted_DSNet_Mask, '*.png'));
GT_Mask = dir(fullfile(Path_ISIC_2017_GT, '*.png'));

Jaccard=0;
Dice=0;
Sensitivity=0;
Specificity=0;
Accuracy=0;

%% Iterate for all the test images
for iter = 1:length(Pred_Mask)
   
  prMask = fullfile(Path_Predicted_DSNet_Mask, Pred_Mask(iter).name);
  gtMask = fullfile(Path_ISIC_2017_GT, GT_Mask(iter).name);

  Current_prMask=imread(prMask);
  Current_gtMask=imresize(imread(gtMask),[size(Current_prMask,1) size(Current_prMask,2)]);
 
  Threshold = isodataAlgorithm(Current_prMask);

  Current_prMask=imbinarize(Current_prMask,Threshold);
  Current_gtMask=imbinarize(Current_gtMask,0);
    
  labelMatrix = bwlabeln(Current_prMask); 
  RegionProp = regionprops(labelMatrix,'Area','Orientation','Extrema');
  maxThreshold = [RegionProp.Area];
  biggestArea = find(maxThreshold==max(maxThreshold));
  Current_prMask(labelMatrix~=biggestArea)=0;

  imwrite(Current_prMask,fullfile(Save_Directory,...
          [Pred_Mask(iter).name(1:12),'_Predict.png']))
  imwrite(Current_gtMask,fullfile(Save_Directory,...
          [Pred_Mask(iter).name(1:12),'_GTruth.png']))
  
  %% Performance Evaluation 
  Jaccard=Jaccard+round(jaccard(Current_prMask,Current_gtMask),4);
  disp(['Image No: ' num2str(iter),' ---> JI = ',...
        num2str(round(jaccard(Current_prMask,Current_gtMask),4))]);
  
  Dice=Dice+dice(Current_prMask,Current_gtMask);
  
  [sen, spe, acc] = Evaluate(Current_gtMask,Current_prMask);
  Accuracy=Accuracy+acc;
  Sensitivity=Sensitivity+sen;
  Specificity=Specificity+spe;
  
  %% Mask overlay with GT where Dice and JI also inserted for evaluation
  position =  [1 38; 184 38];
  Composite_Image = Imfusion(Current_prMask, Current_gtMask);
  Composite_Image_Text = insertText(Composite_Image, position,...
                       [round(dice(Current_prMask,Current_gtMask),3)... 
                       round(jaccard(Current_prMask,Current_gtMask),3)],...
                       'AnchorPoint', 'LeftBottom','FontSize',21);
  imwrite(Composite_Image_Text,fullfile(Save_Directory,... 
          [Pred_Mask(iter).name(1:12),'_Overlaid_GT_Prediction.png']))

%   break;
end

%% Display the avg. metrics
disp('-------------------The Avg. Results---------------')
disp(['Jaccard Index = ',num2str(round(Jaccard/length(Pred_Mask),4))]);
disp(['DSC = ',num2str(round(Dice/length(Pred_Mask),4))]);
disp(['specificity = ',num2str(round(Specificity/length(Pred_Mask),4))]);
disp(['sensitivity = ',num2str(round(Sensitivity/length(Pred_Mask),4))]);
disp(['Accuracy = ',num2str(round(Accuracy/length(Pred_Mask),4))]);

toc;

%% Creates a composite image from two images where Green, Red, and Yellow
% Color respectively indicate TP, FN, and FP.
function compositeImage = Imfusion(Current_prMask, Current_gtMask)
  TP = zeros(length(Current_prMask(:)),1);
  FN = zeros(length(Current_prMask(:)),1);
  FP = zeros(length(Current_prMask(:)),1);
  TN = zeros(length(Current_prMask(:)),1);
  
  flatten_Current_prMask= Current_prMask(:);
  flatten_Current_gtMask= Current_gtMask(:);
  
  for i = 1:1:length(Current_prMask(:))
      if flatten_Current_prMask(i)==1 && flatten_Current_gtMask(i)==1
         TP(i)=1;   
      elseif flatten_Current_prMask(i)==1 && flatten_Current_gtMask(i)==0
          FP(i)=1;
      elseif flatten_Current_prMask(i)==0 && flatten_Current_gtMask(i)==1
          FN(i)=1;
      end
  end
  
  TP = cat(3,reshape(TN,192,256),reshape(TP,192,256),reshape(TN,192,256));
  FN = cat(3,reshape(FN,192,256),reshape(TN,192,256),reshape(TN,192,256));
  FP = cat(3,reshape(FP,192,256),reshape(FP,192,256),reshape(TN,192,256));
  compositeImage = TP+FN+FP;
end

% The function for estimating the sensitivity, specificity, and accuracy.
function [sensitivity, specificity, accuracy] = Evaluate(TrueMask,PredictedMask)
    % Input: TrueMask = Column matrix with actual class labels 
    %        PredictedMask = Column matrix with predicted class labels 
    % Output: sensitivity, specificity, accuracy
    idx = (TrueMask()==1);
    p = length(TrueMask(idx));
    n = length(TrueMask(~idx));
    N = p+n;
    tp = sum(TrueMask(idx)==PredictedMask(idx));
    tn = sum(TrueMask(~idx)==PredictedMask(~idx));
    tp_rate = tp/p;
    tn_rate = tn/n;
    accuracy = (tp+tn)/N;
    sensitivity = tp_rate;
    specificity = tn_rate;
end

function finalThreshold = isodataAlgorithm(grayImage)
    grayImage =  grayImage(:);
    % The itial threshol is equal the mean of grayscale image
    initialTheta = mean(grayImage); 
%     initialTheta = round(initialTheta); % Rounding
    i = 1;
    threshold(i) = initialTheta;
    % Gray levels are greater than or equal to the threshold
    foregroundLevel =  grayImage((grayImage >= initialTheta));
    meanForeground = mean(foregroundLevel(:));
    % Gray levels are less than or equal to the threshold
    backgroundLevel = grayImage((grayImage < initialTheta));
    meanBackground = mean(backgroundLevel(:));
    % Setup new threshold
    i = 2;
%     threshold(2) = round((meanForeground + meanBackground)/2);
    threshold(2) = (meanForeground + meanBackground)/2;
    %Loop: Consider condition for threshold
    while abs(threshold(i)-threshold(i-1))>=0.001
        % Gray levels are greater than or equal to the threshold
        foregroundLevel =  grayImage((grayImage >= threshold(i)));
        meanForeground = (mean(foregroundLevel(:)));
        % Gray levels are less than or equal to the threshold
        backgroundLevel = grayImage((grayImage < threshold(i)));
        meanBackground = (mean(backgroundLevel(:)));
        i = i+1;
        % Setup new threshold
        threshold(i) = round((meanForeground + meanBackground)/2);

    end
    finalThreshold = threshold(end)/2800;
end

%% ------------------------The end-----------------------------------------
