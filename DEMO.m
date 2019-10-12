
clear all; close all;clc


%% load dataset
pathroot       = 'SimpleInput/';
dataset_name   = 'synthetic';
matname        = 0;
data_name      =  ['dataSynth0', num2str(matname),'.mat'];
data_synth_set = [pathroot,'/',data_name];

load(data_synth_set)
data = dataSynth;

param.dataset_name = 'synthetic';
%% set parameters
param.lambda_1   = 0.3;
param.lambda_2   = 0.5;
param.C          = 1;
param.inner_iter = 500;
param.max_iter   = 500;
param.max_inner_iter = 1000;
param.svm = 2; 
param.g   = 0.01;
param.pi_ = 1;
%% run algorithm

fprintf('\n\n----------------------------Algorithm DSL----------------------------\n\n');
tStart = tic;
model = DSL_sdm(data,param);
toc(tStart)
fprintf('\n\n DSL Model Training Is Completed!\n')

%%  visualize the result
fprintf('\n\n--------------------------------Visualize-----------------------------\n\n');
node_idx = model.node_idx;
IDX      =  zeros(100,1);
IDX(node_idx(1:13)) = 1; % PLOT TOP-13 nodes, same number as GT of subgraph
IDX = logical(IDX);


% drawsubgraph(dataSynth.position,dataSynth.idx,dataSynth.in)
% title('Subgraph Groundtruth')

drawsubgraph(dataSynth.position,dataSynth.idx,IDX)
title('Subgraph detected by DSL')


