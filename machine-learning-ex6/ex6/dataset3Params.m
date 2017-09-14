function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma = [0.01,0.03,0.1,0.3,1,3,10,30];

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
error = zeros(length(C),length(sigma));
X1 = X(:,1);
X2 = X(:,2);

for i = 1:length(C)
    for j = 1:length(sigma)
        model = svmTrain(X, y, C(i), @(X1, X2) gaussianKernel(X1, X2, sigma(j)));
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));
    end
end

[Y,col] = min(min(error));
[X,row] = min(error(:,col));

C = C(row);
sigma = sigma(col);


% =========================================================================

end
