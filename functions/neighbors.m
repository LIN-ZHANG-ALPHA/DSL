% Created 2012-Jan-15
% Written by Fu Lin
% Draw the links to indicate neighbors and output the incidence matrix for
% the neighboring nodes.
% Inputs: 
% (1) N   -> the number of nodes
% (2) pos -> the position of nodes
% (3) r   -> the radius to determine neighbors
% Outputs:
% El  -> incidence matrix of neighboring nodes
% idx -> indices of nodes for edges specified in the incidence matrix El

function [El,idx] = neighbors(N,pos,r,ind,pathname)

if nargin < 4
    ind = [];
    pathname = [];
end

% number of edges in complete graph
m = N*(N-1)/2;

% determine the neighbors of each node
idx = zeros(2,m);
stp = 0;

for i = 1 : N
    for j = i+1 : N
        % if the distance between two nodes is less than r
        % then they are neighbors.
        if norm( pos(i,:) - pos(j,:), 2 ) <= r
            stp = stp + 1;
            idx(:,stp) = [i j]';
        end
    end
end

% remove zeros in the indices
idx = idx(:,1:stp);

% construct the incidence matrix for the local outputs
El = sparse(incmat(idx,N));

% draw the graph to indicate neighbors
% drawgraph(pos,idx,ind,pathname)




