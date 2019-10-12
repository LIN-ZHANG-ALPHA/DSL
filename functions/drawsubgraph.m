
function drawsubgraph(pos,idx,in)

% plot the graph where links indicate the neighbors
figure
hold on

% draw the links
ii = sqrt(-1);
for k = 1 : max(size(idx))
    i = idx(1,k);
    j = idx(2,k);
    
    
    % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'r', 'LineWidth', 1 );
    plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'color',[0.82 0.82 0.82], 'LineWidth', 1 );
    
%     if  in(i)&&in(j)
%           plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'b', 'LineWidth', 8 );
%          % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'color',[ 0.8500    0.3250    0.0980], 'LineWidth', 5 );
%          
%     end
    
%     if  in(i)&&~in(j)
%         plot( pos(i,1) + pos(i,2)*sqrt(-1),'y*','LineWidth',2,'MarkerSize',20)
%     end
    
%     if  ~in(i)&&in(j)
%         plot( pos(j,1) + pos(j,2)*sqrt(-1),'y*','LineWidth',2,'MarkerSize',20)
%     end
    
end


for k = 1 : max(size(idx))
    i = idx(1,k);
    j = idx(2,k);
    
    
    % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'r', 'LineWidth', 1 );
    % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'color',[0.82 0.82 0.82], 'LineWidth', 1 );
    
    if  in(i)&&in(j)
         plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'b', 'LineWidth', 2 );
         % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'color',[ 0.8500    0.3250    0.0980], 'LineWidth', 5 );
         % plot( [pos(i,1) + ii*pos(i,2), pos(j,1) + ii*pos(j,2)], 'g', 'LineWidth', 3 ); % gnd 
    end
    
%     if  in(i)&&~in(j)
%         plot( pos(i,1) + pos(i,2)*sqrt(-1),'y*','LineWidth',2,'MarkerSize',20)
%     end
    
%     if  ~in(i)&&in(j)
%         plot( pos(j,1) + pos(j,2)*sqrt(-1),'y*','LineWidth',2,'MarkerSize',20)
%     end
    
end

% draw the nodes
plot( pos(:,1) + pos(:,2)*sqrt(-1),'o','LineWidth',2,'MarkerSize',10)

% draw the selected nodes
IDX =  find(in==1);
plot( pos(IDX,1) + pos(IDX,2)*sqrt(-1),'ks','LineWidth',2,'MarkerSize',15)
% plot( pos(IDX,1) + pos(IDX,2)*sqrt(-1),'gp','LineWidth',2,'MarkerSize',15) % gnd 



% labs = 1:size(pos,1);
% labelpoints(pos(:,1) , pos(:,2),labs); %,'outliers_lin',{'sd', 1.5})

hold off;
h = get(gcf,'CurrentAxes');
set(h, 'FontName', 'cmr10', 'FontSize', 18)









