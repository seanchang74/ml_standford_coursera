function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possibles = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
bestC = possibles(1);
bestSigma = possibles(1);
Err = 1000;  % just set error to a very large value

for i=1:length(possibles)
C_para = possibles(i);
	for j=1:length(possibles)
	Sigma_para = possibles(j);
		% using new parameters to train new svm model
		model = svmTrain(X, y, C_para, @(x1, x2) gaussianKernel(x1, x2, Sigma_para));
		% calculate the error in new train model
		predictions = svmPredict(model, Xval);
		pred_err = mean(double(predictions ~= yval));
		if pred_err < Err
			bestC = C_para;
			bestSigma = Sigma_para;
			Err = pred_err;
		end;
	end;
end;

C = bestC;
sigma = bestSigma;


% =========================================================================

end
