
This toolbox provides a discriminative Subgraph learning algorithm developed in the paper:
Lin Zhang and Petko Bogdanov,
"DSL: Discriminative Subgraph Learning via Sparse Self-Representation" (Published in SDM 2019)


To execute, run the DEMO.m file:
     
	 main function: 
	 model = DSL_sdm(data, param)
	 
    % Input:
    %     + data: struct data type
    %           .X   : dataset X  with size as [num_Feature num_Sample]
    %           .gnd : class labels with size as [1 num_Sample]
    %           .L   : laplacian matrix with size as [num_Feature num_Feature]
    %     + param: structure
    %           .lambda_1:  L21 tradeoff for sparsness
    %           .lambda_2:  trace-norm tradeoff for smoothness
    %           .pi_:       tradeoff for SVM part in model
    %           .C:         svm hyperparamter 
    %           .g:         svm hyperparamter 
    %           .max_iter:  max iteration for outside loop
    %           .max_inner_iter:   max iteration for nest inner loop
    %           .inner_iter:       max iterations for quadratic programming
    % Output:
    %     + model: struct data type
    %           .w: w in the svm
    %           .b: b in the svm
    %           .Phi: coeff to form each dim of transformed space
    %           .node_idx: indices of selected nodes/features

	 
     
If you are using our code for your research please cite:
"DSL: Discriminative Subgraph Learning via Sparse Self-Representation", 
Lin Zhang and Petko Bogdanov, SDM2019


