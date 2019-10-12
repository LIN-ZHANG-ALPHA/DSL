
function  M = cal_M(n,d,w,X)


M = zeros(n,d);
% for i  = 1 : n
%     tt = X(:,i)*w';
%     M(i,:) = tt(:);
% end


for i  = 1 : n
    tt = X(:,i)*w';
    M(i,:) = tt(:);
%     M(i,:) = reshape(X(:,i)*w',[],1);
end



% K1 = zeros(n,n);
% for i  = 1 : n
%    
%     Temp = w*X(:,i)';
%     U    = Temp(:)';
%     K1(i,:)  = U;
%     U = [];
% end
% 
% kk =1;
%% solution 2

% for i  = 1 : n
%     tt = w * X1(i,:);
%     M(i,:) = tt(:);
% %     M(i,:) = reshape(X(:,i)*w',[],1);
% end

% [sw,sh] = size(X);
% T02 = repelem(X',1,sw);
% 
% % 
% T_2 = repmat(w,1,sw);
% % % T_2 =  bsxfun(@times, w, ones(sw,sw));
% T_2 = T_2(:);
% T_2_mat = repmat( T_2, 1, sh );
% T = T02.*T_2_mat';

% 
% M1 = T02.*T_2_mat';

