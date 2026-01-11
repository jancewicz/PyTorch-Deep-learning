import torch

"""
Autograd (automatic gradients) - implementation of reverse-mode autodifferentiation.

Consider f(x) = x**2, derivative of this function is f'(x) = 2x

f(5) = 25 and f'(5) = 10
"""

# requires_grad is telling that this is variable. Knowing this PyTorch is going to keep track of all operations with x
x = torch.tensor(5.0, requires_grad=True)
f = x**2  # tensor(25., grad_fn=<PowBackward0>)

# Backpropagates the gradients through the computation graph all the way to the leaf node (x in this case)
f.backward()

# We can read grad attribute which was computed during backprop. This gives us derivative of f with regard to x
grad = x.grad  # tensor(10.)

"""
After computing gradients, gradient descent step is made. To do this is necessary to disable temporarily gradient
tracking since you don't want to track gradient descent steps in the computational graph. 
"""
learning_rate = 0.1
with torch.no_grad():
    x -= (
        learning_rate * grad
    )  # gradient descent step, variable x gets decremented by 0.1 * 10 = 1, down from 5. to 4.

# Lastly, it's essential to zero out the gradients of every model parameter
x.grad.zero_()

# Whole training loop should have this structure
learning_rate = 0.1
x = torch.tensor(5.0, requires_grad=True)
for epoch in range(100):
    f = x**2  # forward pass
    f.backward()  # backward pass
    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero()
