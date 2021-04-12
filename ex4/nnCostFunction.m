function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	% Convert y vector to matrix of rows of size 10 %
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%	yMatrix = zeros(length(y),num_labels);
%%	for i = 1:length(y)
%%		yMatrix(i,y(i)) = 1;
%%	end
%%	%yMatrix(1:10,:)
%%	%yMatrix(4990:5000,:)
%%
%%	%%%%%%%%%%%%%%%%%%%%%%%
%%	% Add bias units to X %
%%	%%%%%%%%%%%%%%%%%%%%%%%
%%
%%	X = [ones(m,1) X];
%%
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	% Vectorized Forward Propagation %
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	a1 = X';
%%	z2 = Theta1 * a1;
%%	a2 = sigmoid( z2 );
%%	a2 = [ones(m,1) a2']';
%%	z3 = Theta2 * a2;
%%	a3 = sigmoid( z3 );
%%
%%	%%%%%%%%%%%%%%%%%
%%	% Cost function %
%%	%%%%%%%%%%%%%%%%%
%%
%%	%% Loop version
%%
%%%	J = 0;
%%%	for i = 1:m	
%%%		J += yMatrix(i,:) * log(a3(:,i)) + (1-yMatrix(i,:)) * log(1-a3(:,i));
%%%	end
%%%
%%%	J *= -(1/m)
%%
%%	%% Vectorized version
%%
%%	J = -(1/m) * sum( sum( yMatrix .* log(a3)' + (1-yMatrix) .* log(1-a3)',2 ) );
%%
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	% Regularization of the cost function %
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%%	%% Loop version 	
%%%
%%%	thetaSumSqr = 0;
%%%	for i = 2:input_layer_size+1
%%%		for j = 1:hidden_layer_size
%%%			thetaSumSqr += Theta1(j,i).^2;
%%%		end
%%%	end
%%%	for i = 2:hidden_layer_size+1
%%%		for j = 1:(num_labels)
%%%			sizeSS = size(thetaSumSqr);
%%%			thetaSumSqr += Theta2(j,i).^2;
%%%		end
%%%	end
%%%	thetaSumSqr;
%%%	J += (lambda / (2*m)) * thetaSumSqr;
%%
%%	%% Vectorized version
%%
%%	J += lambda/(2*m) * ( sum(sum( Theta1(:,2:end).^2,2 )) + sum(sum( Theta2(:,2:end).^2,2 )) );
%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	% Backpropagation algorithm %
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	
%%%	a1 = X;
%%%
%%%	for t = 1:m		
%%%		a1t = a1(t,:); 
%%%		z2t = Theta1 * a1t';
%%%		a2t = sigmoid( z2t );
%%%		z3t = Theta2 * [1; a2t];
%%%		a3t = sigmoid( z3t );
%%%		
%%%		d3 = a3t - yMatrix(t,:)';
%%%		d2 = (Theta2' * d3)  .* [1; sigmoidGradient(z2t)];
%%%		d2 = d2(2:end); % remove bias activation unit (optional)
%%%
%%%		Theta1_grad += d2 * a1t;
%%%		Theta2_grad += d3 * [1; a2t]';
%%%
%%%	end
%%
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%	% Vectorized Backpropagation algorithm %
%%	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%	d3 = a3 - yMatrix';	
%%	d2 = (Theta2' * d3) .* a2 .* (1-a2);	% OK
%%	%d2 = (Theta2' * d3) .* [ones(1,m); sigmoidGradient(z2)]; %OK aussi
%%	
%%	Theta1_grad = d2 * a1';	
%%	Theta2_grad = d3 * [ones(1,m); a2]';	
%%
%%	Theta1_grad = Theta1_grad(2:end,:);
%%	Theta2_grad = Theta2_grad(:,2:end);
%%
%%	%%%%%%%%%%%%%%%%%%
%%	% Regularization %
%%	%%%%%%%%%%%%%%%%%%
%%
%%	Theta1_grad = (1/m) * ( Theta1_grad + ( lambda * [ zeros(size(Theta1,1), 1) Theta1(:,2:end) ] ));
%%	Theta2_grad = (1/m) * ( Theta2_grad + ( lambda * [ zeros(size(Theta2,1), 1) Theta2(:,2:end) ] ));
%%%	%% on ajoute une ligne de 0 devant le reste des thetas pour palier à la condition j = 0 or j != 0 pour la régularization		

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Entrainement Numero 2 --> on refait tout sans regarder pour bien intégrer %
	% et on creuse chaque ligne pour optimizer et bien comprendre %

	%%%%%%%%%%%%%%%%
	% y to yMatrix %
	%%%%%%%%%%%%%%%%

	%yM = zeros(num_labels,m);
	%for i = 1:m
	%	yM(y(i),i) = 1;
	%end
	yM = eye(num_labels)(:,y');

	%%%%%%%%%%%%%%%%%%%%%%%
	% Forward propagation %
	%%%%%%%%%%%%%%%%%%%%%%%
	
	a1 = [ ones(1,m); X' ];
	z2 = Theta1 * a1;
	a2 = [ ones(1,m); sigmoid( z2 ) ];
	z3 = Theta2 * a2;
	a3 = sigmoid( z3 );

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Vectorized Cost function %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	J = (-1/m) * sum( sum( yM .* log(a3) + (1-yM) .* log(1-a3) ,2) );

	%%%%%%%%%%%%%%%%%%%%%%%
	% Cost Regularization %
	%%%%%%%%%%%%%%%%%%%%%%%

	J += (lambda/(2*m)) * ( sum(sum( Theta1(:,2:end).^2,2)) + sum(sum( Theta2(:,2:end).^2,2)) );
	
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Backpropagation algorithm %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	d3 = a3 - yM;	
	d2 = (Theta2' * d3) .* a2 .* (1-a2);
	d2 = d2(2:end,:); % Remove bias unit herited from Theta2

	Delta1 = d2 * a1'; 
	Delta2 = d3 * a2';

	Theta1_grad = (1/m) * ( Delta1 + (lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)] ));
	Theta2_grad = (1/m) * ( Delta2 + (lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end)] ));
	% on ajoute une ligne de 0 devant le reste des thetas pour mettre en place la condition
	% de regularization j = 0 or j != 0
	% ça empeche de regularizer le bias term

	% OK 100% --> mieux construite que la précèdente version !

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
