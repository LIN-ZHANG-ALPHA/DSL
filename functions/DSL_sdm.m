
function [model] = DSL_sdm(data, param)
% refer: Discriminative Subgraph Learning via Max-Margin Matrix Decomposition

%% Problem:
% min_W,b,Phi ||X'-X'*Phi||^2 + lambda1 * ||Phi||_21+lambda2 tr(Phi' L Phi)+ SVM

%% Input and output:
% Input:
%     + data: struct data type
%           .X [nFea nSmp]: dataset X (no need?)
%           .gnd [1 nSmp]: class labels
%           .L [nFea nFea]: laplacian matrix
%     + param: structure

%           .lambda_1:  L21 tradeoff for sparsness
%           .lambda_2:  trace-norm tradeoff for smoothness

%           .pi_:    tradeoff for SVM part in model
%           .C:   svm hyperparamter 
%           .g:   svm hyperparamter 


%           .max_iter: max iteration for outside loop
%           .max_inner_iter:   max iteration for nest inner loop
%           .inner_iter:   max iterations for quadratic programming

% Output:
%     + model: struct data type
%           .w: w in the svm
%           .b: b in the svm
%           .Phi: coeff to form each dim of transformed space
%           .node_idx: indices of selected nodes/features
% (c) 2018  Lin Zhang UAlbany
%------------------------------------------------------------------------%



%% initialization
if (~exist('eps', 'var'))
    eps = 1e-8;
end
if (~exist('eta', 'var'))
    eta = 0.999; % 0.999
end


if nargin < 2
    lambda_1 = 0.15;
    lambda_2 = 0.1;
    C = 1;
    max_iter = 500;
    g = 0.1;
    pi_ = 1;
else
    lambda_1 = param.lambda_1;
    lambda_2 = param.lambda_2;
    max_iter = param.max_iter;
    C =  param.C;
    % g =  param.g;
    pi_ = param.pi_;
end


%% load data and preparing
dataset_name = param.dataset_name;

switch dataset_name

    case 'synthetic'
        
        %  synthetic data
        X = data.X; % features
        X = Xnorm(X,1);
        y = data.gnd;
        
        
        train_IDx = [1:150,201:350];
        Xtrain = X(:,train_IDx);
        [m,n]  =  size(Xtrain); % [nFea nSmp] = size(data.X);
        ytrain =  y(train_IDx);
        
        valid_IDX = [151:200,351:400];
        Xvalid = X(:,valid_IDX);
        yvalid = y(valid_IDX);
        
    otherwise
        fprintf('Error, no such data is found! Try again!\n')
end




%%
if isfield(data,'L')
    L = data.L; % Laplacian Matrix
else
    Adj = data.W;
    Degree = diag(sum(Adj, 2));
    L = Degree - Adj; % Laplacian Matrix
end
% L = eye(m,m) - data.W;
tic
%% main loop
iter    = 0;
obj_old = 0;
% Phi     = zeros(m,m);
Phi     = eye(m,m);
% Phi     = 0.0001*randn(m,m);
while iter < max_iter
    
    iter =  iter+1;
    
    % ++++++++++++++++++++++++++++  update {w,b} using LIBSVM
    if iter > 1
        Xtrain_proj =  Phi'* Xtrain; % update the feature
        Xvalid_proj =  Phi'* Xvalid;
    else
        Xtrain_proj =  Xtrain;
        Xvalid_proj =  Xvalid;
    end
    
    if ~isfield(param,'svm')||param.svm == 2
        % classifier = svmtrain(ytrain', Xtrain_proj', sprintf('-s 0 -t 0 -c %g -g 0.01', C));
        if isfield(param,'g')
            g = param.g;
            classifier = svmtrain(ytrain', Xtrain_proj', sprintf('-s 0 -t 0 -c %g -g %g', C,g));
        else
            classifier = svmtrain(ytrain', Xtrain_proj', sprintf('-s 0 -t 0 -c %g', C));
        end
        
        w =  classifier.SVs' * classifier.sv_coef;
        b =  -classifier.rho;
        % [lbl, acc, dec] = svmpredict(yvalid', Xvalid_proj', classifier, []);
        
    elseif param.svm == 1  % 1-norm svm
        classifier = train(ytrain', sparse(Xtrain_proj'), '-c 1 -s 5 ');
        
        w = classifier.w + eps;
        w = w';
        b = classifier.bias;
        % [lbl, acc, dec] = predict(yvalid', sparse(Xvalid_proj'), classifier);
    end
    
    
    % ++++++++++++++++++++++++++++  update {\Phi}
    % Phi = update_Phi(Xtrain,ytrain,w,b,L,param);
    
    Phi = update_Phi_v1(Xtrain,ytrain,w,b,L,param);
    % Phi = update_Phi_oldversion(Xtrain,ytrain,w,b,L,param);
    
    Phi =  Phi-diag(diag(Phi));
    % ++++++++++++++++++++++++++++ stop condition check
    [obj_new,obj_mag]  = getObj(w, b, Xtrain_proj, ytrain, C, L, Phi,lambda_1, lambda_2,pi_);
    
    residual = abs(obj_old-obj_new); %/obj_new;
    disp(['obj-',num2str(iter),'=',num2str(obj_new),',','residual-',num2str(iter),'=',num2str(residual)]);
    if residual <  10e-3
        break;
    else
        obj_old = obj_new;
    end
end

% get the selected node ID
% Phi =  Phi-diag(diag(Phi)); % avoid self-expression
[dumb, idx]    = sort(sum(Phi.*Phi,2), 'descend');  % l21 norm
model.node_idx = idx;

model.Phi = Phi;
model.w   = w;
model.b   = b;
model.obj_mag = obj_mag;
model.param   = param;
toc
end

function [Obj,obj_mag] = getObj(w, b, X, y, C, L, Phi,lambda_1, lambda_2,pi_)
if nargin < 10
    pi_ = 1;
end

% min  ||X'-X'Phi||_{F,2} + lambda1 * ||Phi||_{2,1} + lambda2 *Tr(Phi'LPhi) + svm

Term_1 = X'-X'*Phi;

% obj_nmf = norm(Term_1,'fro')+ lambda_1* L21Norm(Phi) + lambda_2*trace(Phi'*X*Phi);
fitting_term = sum(sum(Term_1.*Term_1));
sparse_term  = L21Norm(Phi);
smooth_term  = trace(Phi'*L*Phi);
obj_nmf = fitting_term + lambda_1* sparse_term + lambda_2*smooth_term;

% f = 0.5*lambda*sum(w.^2) + 2*sum(max(0, ell-y'.*(w'*X)));
obj_svm = 0.5 * sum(w.^2) + C * sum(max(0, 1 - y .* (w'*X + b)));

Obj = pi_*obj_svm + obj_nmf;


if nargout > 1
    obj_mag.fitting_term = fitting_term;
    
    obj_mag.sparse_term = [lambda_1,sparse_term];
    obj_mag.smooth_term = [lambda_2,smooth_term];
    obj_mag.svm_term = [pi_,obj_svm];
end

end
