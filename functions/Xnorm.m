function [Y]= Xnorm(X,opt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% Normalize each feature (row) of X to have 0-mean and (optional) 1-std.dev. 
%
% INPUT ARGUMENTS:
%   X: DxN matrix of original data in columns
%   opt: not empty then 1-std dev (optional)
% OUTPUT ARGUMENTS:
%   Y: DxN normalized data
% (c) 2010 QuangAnh Dang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



Y=bsxfun(@minus,X,mean(X,2)); %-- subtract each X's col from the mean

% [D,N]=size(X); 
% Y = X-repmat(mean(X')',1,N); %-- 0-mean (repmat: create N mean-cols)
% Y = Y/max(max(abs(Y)));      %-- required for spec. clustering!

%-- one-stand. dev.
if nargin>1
    s = std(Y,0,2);
    s((s<eps))=1; %-- set stddev = 1 for "constant value" features
    Y = bsxfun(@rdivide,Y,s);
    
    %Y = Y./repmat(std(Y')',1,N); 
end
