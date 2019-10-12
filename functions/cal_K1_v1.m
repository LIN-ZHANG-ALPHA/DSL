
% function K1 = cal_K1(w,X,M,t01,n,d)
%  function K1 = cal_K1_v1(M,K0,t01)


function K1 = cal_K1_v1(M,Xt,K0,t01)


% function K1 = cal_K1_v1(n,d,w,X,M,K0,t01)
% CREATE: 09/25/2018
% update 09/26/2018
% update 10/05/2018



%% solution 1
[a,b,c] = size(K0);

sx = size(K0);
sy = size(t01);
dx = ndims(K0);
% Xt = reshape(permute(K01, [1 3:dx 2]), [prod(sx)/sx(2) sx(2)]); 

% t01(abs(t01) < 8*1e-3) = 0; % threshold to sparse

% t01 = sparse(t01);
% Z2 = mtimesx_sparse(Xt,'N',t01,'N');
% Z2 =  ssmult(Xt,t01) ;
Z2 = Xt * t01;

% Z2 = full(Z2);

Z2 = permute(reshape(Z2, [sx([1 3:dx]) sy(2)]), [1 dx 2:dx-1]);

U1 = reshape(Z2,[a,b*c]);

% U1 = sparse(U1);
K1 =  M*U1';


