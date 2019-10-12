
function  [M,K0,Xt] = pre_comp(n,d,w,X)


% update 10/05/2018

[a,b] = size(X);
%% solution 1
% for i  = 1 : n
%     tt = X(:,i)*w';
%     M(i,:) = tt(:);
%     
%     % tt1 = tt';
%     K0(i,:,:)  = tt';
% end


%% solution 2

T02 = repmat(X',1,a)';

T03 = repelem(w',a,1);
T03 = T03(:);
T_03_mat = repmat( T03, 1, b );

M  = T02'.*T_03_mat';

% M(abs(M) < 8*1e-3) = 0; % % threshold to sparse

K =  reshape(M,[n,d^0.5,d^0.5]);

K0 = permute(K, [1 3 2]);% Y = zeros(m, n, p);

% sx = size(K0);
% dx = ndims(K0);
% Xt = reshape(permute(K0, [1 3:dx 2]), [prod(sx)/sx(2) sx(2)]);

Xt =  reshape(K,[a*b,a]);


% Xt = sparse(Xt);
% M  = sparse(M);


