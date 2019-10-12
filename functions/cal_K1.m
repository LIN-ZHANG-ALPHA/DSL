
function K1 = cal_K1(w,X,M,t01,n,d)
% function K1 = cal_K1(w,X,M,t01)
% CREATE: 09/25/2018
% update 09/26/2018

K1 = zeros(n,n);
for i  = 1 : n
    % P0(i,:,:) = w*X(:,i)'*t01;
    Temp = w*X(:,i)'*t01;
    U    = Temp(:)';
    K1(i,:)  = ( M*U');
    U = [];
end

%%
% T01  = X'*t01;

%% % solution 1:

% K1_1   = zeros(n,d);
% for i  = 1 : n
%     Temp       = w * T01(i,:);
%     K1_1(i,:)  = Temp(:);
% end
% 
% K1 = M*K1_1';


%%  solution 2:

% [sw,sh] = size(X);
% T02 = repelem(T01,1,sw);
% % clear T01
% T_2 = repmat(w,1,sw);
% % T_2 =  bsxfun(@times, w, ones(sw,sw));
% T_2 = T_2(:);
% T_2_mat = repmat( T_2, 1, sh );
% % T_2_mat = bsxfun(@times, T_2, ones(sw*sw,sh));
% 
% % clear T_2
% 
% T = T02.*T_2_mat';
% % T = bsxfun(@times,T02,T_2_mat');
% % T = mtimesx(T02,'N',T_2_mat,'T'); 
% % clear T02 T_2_mat
% K1 = M*T';

% K1 = mtimesx(M,T'); 
% mtimesx('SPEEDOMP','OMP_SET_NUM_THREADS(4)') % sets SPEEDOMP mode with number of threads = 4

%%  solution 3:

% [sw,sh] = size(X);
% T02 = repelem(T01,1,sw);
% % clear T01
% T_2 = repmat(w,1,sw);
% T_2 = T_2(:);
% % T_2_mat = repmat( T_2, 1, sh );
% % clear T_2
% for i = 1:sh
%     T(:,i) = T02(i,:).*T_2';
% end
% 
% clear T02 T_2_mat
% K1 = M*T;




