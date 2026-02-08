import torch.nn as nn

"""
One of key insights from paper "Understanding the difficulty of training deep feedforward neural networks -
Glorot, X. & Bengio, Y. (2010)" was that the problem with unstable gradients happens because of poor choice of
activation functions.

Sigmoid fn -> ReLU: Rectified linear unit - f(x) = max(0, x)
ReLU does not saturate for positive values, really fast to compute.

But ReLU suffers from problem od dying neurons. A neuron dies if its weights are tweaked in a way that the input of Relu
function is negative for all instances in training set. In this scenario, functions just keeps outputting zeros and
gradient descent does not affect it anymore.

The solution to this problem:
ReLU -> LeakyReLU: f(z) = max(α * z, z)
Hyperparameter alpha decides how much the function "leaks". This creates small slope for all values z < 0, and protect 
neurons from dying. 

Other variants of this activation function:
RReLU - Randomized leaky ReLU - alpha is picked randomly from given range during training, then fixed to average value 
when testing. It seemed to work as a regularizer, decreasing the risk of model overfitting.

PReLU - Parametric leaky ReLU - alpha is learned during the training. Instead of being the hyperparameter, it can be 
tweaked by backpropagation like other parameters. It was outperforming ReLU on huge datasets, but with a risk of 
overfitting training sets in smaller datasets 
"""

alpha = 0.2  # declare alpha value
model = nn.Sequential(
    nn.Linear(50, 40), nn.LeakyReLU(negative_slope=alpha)
)  # leakyReLU with slope alpha
nn.init.kaiming_uniform_(
    model[0].weight, alpha, nonlinearity="leaky_relu"
)  # apply leakyReLU to initialization of model weights
