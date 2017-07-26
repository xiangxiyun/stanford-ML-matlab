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
% Theta1 size: hidden_layer_size* (input_layer_size + 1)

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Theta2 size: output_layer_size* (hidden_layer_size + 1)

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 * 401
Theta2_grad = zeros(size(Theta2)); % 10 * 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X]; % size: #input * (input_layer_size+1) 5000*401
z2 = a1*Theta1';
a2 = [ones(m, 1) sigmoid(z2)]; % size: #input * (hidden_layer_size_1) 5000*26
z_3 = a2*Theta2';
h = sigmoid(z_3); % size: #input * ouput_layer_size 5000*10

all_combos = eye(num_labels);  
y_matrix = all_combos(y,:) ; % size: 5000*10

for i = 1:num_labels
    J = J + (y_matrix(:, i)'*log(h(:, i)) + (1-y_matrix(:, i)')*log(1-h(:,i)));
end

J = -1/m*J + lambda/(2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));


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

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m
    a_1 = [1 X(t, :)]; % size 1*(m+1)  1*401
    z_2 = a_1 *Theta1'; % size 1* hidden_layer_size 1*25
    a_2 = [1 sigmoid(z_2)]; % size 1*(hidden_layer_size + 1)  1*26
    z_3 = a_2*Theta2'; %size 1*output_layer_size  1*10
    a_3 = sigmoid(z_3); %size 1*output_layer_size 1*10
    
    Deta3 = (a_3 - y_matrix(t, :))';  % size 10*1
    Deta2 = Theta2(:, 2:end)' * Deta3 .* sigmoidGradient(z_2)'; %size 25*1
    
    D1 = D1 + Deta2* a_1; % size: 25* 401
    D2 = D2 + Deta3* a_2; % size: 10* 26
end

Theta1_grad = D1/m;

Theta2_grad = D2/m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m *Theta1(:, 2:end);

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m *Theta2(:, 2:end);
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
