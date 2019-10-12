
function Phi = update_Phi(X,y,w,b,L,param)

% Input:
%  y   - Label vector {-1,+1}  [n x 1]
%  X   - Feature matrices      [m x n], n is the number of
%  instance; d are the size of features
%  w   - svm , w
%  b   - svm , b
%  L:  - Laplcian matrix
% Output:
%     + Phi: node selection matrix


% (c) 2018  Lin Zhang @UAlbany
% updated on 09/25/2018

%%
if nargin > 3
    C          = param.C;
    C_         =  param.C*param.pi_;
    inner_iter = param.inner_iter;
    maxIter    = param.max_inner_iter;
    lambda_1   = param.lambda_1;
    lambda_2   = param.lambda_2;
    
else
    C          = 1;
    C_         = C*pi;
    inner_iter = 100;
    maxIter    = 100;
    lambda_1   = 0.15;
    lambda_2   = 0.1;
    
end

%%

XX = X*X';

[m,n]  = size(X); % n: number of samples; m: num. nodes
d      = m^2; % feature dim square: d = m x m

% % M0 = zeros(n,m,m);
% M = zeros(n,d);
% for i  = 1 : n
%     tt = X(:,i)*w';
%     M(i,:) = tt(:);
% end

M = cal_M(n,d,w,X);

% M = reshape(M0, [n, d]); % n x (m^2)
% P = reshape(M1, [n, d]);

% [Ix,Iy] =  size(y);
% if Ix == 1
%     y = y';
% end


% D       = eye(m,m);
% Phi     = eye(m,m);
% Phi_old = eye(m,m);


% Z_partial_inv = 0.5 * (XX + lambda_2*L)^(-1);

%% ++++++++++++++++++++++++++++ main loop
iter = 1;
while iter < maxIter
    
    % ++++++++++++++++++++++++++++ update D with previous Phi
    if iter > 1
        % need to conside the zero-rows when taking gradient, eps is to aviod all-zeros case
        D = diag(1./sqrt(sum(Phi_old.*Phi_old,2)+eps))*0.5;
        %         d = sqrt(sum(W.*W,2)+eps);
        %         D = diag(0.5./d);
    else
        D       = eye(m,m);
    end
    
    % ++++++++++++++++++++++++++++ Update Z, R, A1, A2
    % Z =  0.5*(XX^T+lambda_1 * D+ lambda_2 * L)^-1
    
    Z  = 0.5 * (XX + lambda_1*D+ lambda_2*L)^(-1);
    % Z =  Z_partial_inv + 0.5 * (lambda_1*D)^(-1);  %
    
    
    % R = XX^T+lambda_1 * D+ lambda_2 * L
    R = XX + lambda_1*D - lambda_2*L;
    
    % G =  tr(XX^T)-2tr(XX^T Z XX^T)-2tr(TXX^T Z^T XX^T)+4tr(XX^T Z^T R Z XX^T)
    % G =  trace(temp1) -2*trace(temp1*Z*temp1)-2*trace(temp1*Z'*temp1)+ 4*trace(temp1*Z'*R*Z*temp1);
    
    % ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % ++++++++++++++++++++++++++++ using box constraint quadratic programming
    % A1 = tr(w X_i^T Z^T R Z X_j w^T) - w^T( w X_i^T Z)X_j
    RZ = R*Z; % Update 09/25/18
    
    % t01  = Z'*R*Z;
    t01 = Z'*RZ;
    %     % P0 = zeros(n,m,m);
    % U = sparse(n,d);
%     U = zeros(n,d);
%     for i  = 1 : n
%         % P0(i,:,:) = w*X(:,i)'*t01;
%         Temp = w*X(:,i)'*t01;
%         U(i,:) =  Temp(:);
%     end
%     % U   = reshape(P0, [n, d]);
    
    % clear P0
    
    % K1  = ( M*U').* (y*y');
    
    % K1 = cal_K1(w,X,M,t01);
    K1 = cal_K1(w,X,M,t01,n,d);
    %         K1 = zeros(n,n);
    %         for i  = 1 : n
    %             % P0(i,:,:) = w*X(:,i)'*t01;
    %             Temp = w*X(:,i)'*t01;
    %             U    =  Temp(:)';
    %             K1(i,:)  = ( M*U');
    %             U = [];
    %         end
    
    
    A = y*y';
    if size(A,1)==1
        A = y'*y;
    end
    
    K1 =  K1.* A;%  K1.* (y*y');
    
    t02  = w'* w; % this is scalar, thus X_i*w'*w = t1*X_i;
    P1   = Z * X;
    % K2   = -(t02*P1'*X).* (y*y'); % yiyj(- w^T *w *X_i^T *Z *X_j)
    K2   = -(t02*P1'*X).* A; % yiyj(- w^T *w *X_i^T *Z *X_j)
    K    = 2*(K1 + K2); % by 2 is because: min 0.5*x'*H*x + f'*x  in libqp_gsmo
    
    clear K1 K2
    % ++++++++++++++++++++++++++++ A2, for one alpha
    t10 = XX*Z';
    t11 = XX*Z;
    % t12 = Z'*R*Z;
    t12 = Z'*RZ;
    
    % f0  = 1-b*y-2*y'.*(w'*t10*X); % 1-yib-2yi w'(X X' Z')xi
    f0   = 1-b*y-2*(w'*t10*X).*y;  % 1-yib-2yi w'(X X' Z')xi
    
    t13  = t11 - 2*XX*t12;      % XX'Z-2XX'Z'RZ
    t130 = reshape(t13',[d,1]); % transpose is Necessary, this will make tr() as inner product
    f1   = -(M * t130).*y';    % -yi Tr[(X*X'*Z*xi*w'- 2X*X'*Z'*R*Z)*xi*w']
    
    t14  = t11' - 2*t12*XX;
    t140 = reshape(t14,[d,1]);
    % here give w*xi' transpose,which is M, not P, to make tr()  as inner product
    f2   =  - (M * t140).* y'; % -yi Tr[w x' (Z'XX'-2Z'RZXX')]
    
    f = f0 + f1' + f2';
    
    clear f0 f1 f2
    % box constraint quadratic programming to solve the alpha
    opt       = struct('TolKKT', eps/100, 'MaxIter', inner_iter, 'verb', 0);
    LB        = zeros(n,1);
    % UB        = C * ones(n,1);
    UB        = C_ * ones(n,1);
    [alpha,~] = libqp_gsmo(-K, f, y', 0, LB, UB, [], opt); % % call package here ( taking negative)
    
    % ++++++++++++++++++++++++++++ update Phi
    h   = M'*(alpha.*y'); % (d x n) x (n x 1) = d x 1
    H   = reshape(h, [m, m]); % d = m x m
    Phi = Z*(H + 2*XX); % M_i = xi*w', XX =  X*X';
    % Phi = (0.5 * (X*X' + lambda_1*D+ lambda_2*L))\(H + 2*XX); % M_i = xi*w', XX =  X*X';
    
    
    % ++++++++++++++++++++++++++++ check if D is coverged
    
    if iter > 2
        obj_inner = norm(Phi-Phi_old);
        if obj_inner <  1e-1 % 1e-5
            fprintf('inner iter = %d, obj_inner = %d\n', iter, obj_inner);
            break;
        else
            Phi_old = Phi;
        end
    else
        Phi_old = Phi;
    end
    
    iter = iter+1;
    
    if mod(iter, 100) == 0
        fprintf('inner iter = %d, obj_inner = %d\n', iter, obj_inner);
    end
    % clear K f Z R
end






