import torch
from fundamentals.linear_regression.data import X_train, y_train, X_test

"""
Creating simple linear regression model with low level pytorch api utilities.

This if first part of comparing low level PyTorch api and higher level tools. The data for this program was prepared
inside data.py file using sklearn train_test_split method.
"""

# Create parameters for linear regression
torch.manual_seed(42)
n_features = X_train.shape[1]  # 8 input features
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)

# Train model
if __name__ == "__main__":
    learning_rate = 0.4
    n_epochs = 20
    for epoch in range(n_epochs):
        y_pred = X_train @ w + b
        loss = ((y_pred - y_train) ** 2).mean()  # mean squared error loss
        loss.backward()  # compute the gradients of the loss with regard to every parameter
        with torch.no_grad():
            b -= learning_rate * b.grad
            w -= learning_rate * w.grad
            b.grad.zero_()
            w.grad.zero_()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

    # After model training we can try to predict unseen data
    X_new = X_test[:3]
    with torch.no_grad():
        y_pred = X_new @ w + b

    print(y_pred)


