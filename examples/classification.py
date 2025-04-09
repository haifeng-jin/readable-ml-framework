import numpy as np
from sklearn.model_selection import train_test_split

import framework
from framework import ops

# Set the random seed for reproducibility
np.random.seed(42)


# Generate synthetic data (same as the NumPy example)
n_samples = 50
centers = [[1, 1], [5, 3], [8, 0]]
n_classes = len(centers)
X_np = np.zeros((n_samples * n_classes, 2), dtype=np.float32)
y_np = np.zeros(n_samples * n_classes, dtype=np.int64)
for i, center in enumerate(centers):
    X_np[i * n_samples : (i + 1) * n_samples, :] = (
        center + np.random.randn(n_samples, 2) * 0.8
    )
    y_np[i * n_samples : (i + 1) * n_samples] = i

# One-hot encode y
y_np = np.eye(n_classes)[y_np].astype(np.float32)  # One-hot

# Shuffle the data
indices = np.arange(len(X_np))
np.random.shuffle(indices)
X_np = X_np[indices]
y_np = y_np[indices]

# Split data into training and testing sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

# Convert NumPy arrays to framework tensors
X_train = framework.Tensor.from_numpy(X_train_np)
X_test = framework.Tensor.from_numpy(X_test_np)
y_train = framework.Tensor.from_numpy(y_train_np)
y_test = framework.Tensor.from_numpy(y_test_np)


class Module:
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def parameters(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, input_size, output_size):
        weight = (np.random.randn(input_size, output_size) * 0.01).astype(
            np.float32
        )
        bias = np.zeros((1, output_size)).astype(np.float32)
        self.weight = framework.Tensor.from_numpy(weight)
        self.bias = framework.Tensor.from_numpy(bias)

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        x = ops.matmul(x, self.weight)
        x = ops.add(x, self.bias)
        return x


# Define the MLP model
class SimpleMLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = ops.relu(x)
        x = self.fc2(x)
        x = ops.softmax(x)
        return x


def categorical_cross_entropy(y_true, y_pred):
    batch_size = y_true.shape[0]
    loss = ops.multiply(y_true, ops.log(y_pred))
    loss = ops.sum(loss)
    loss = ops.multiply(loss, -1.0 / batch_size)
    return loss


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        for parameter in self.parameters:
            ops.add_(parameter, ops.multiply(parameter.grad, -self.lr))


# Instantiate the model
input_size = 2
hidden_size = 10
output_size = n_classes  # Number of classes
model = SimpleMLP(input_size, hidden_size, output_size)

# Define loss function and optimizer
optimizer = SGD(model.parameters(), lr=0.05)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = categorical_cross_entropy(outputs, y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.numpy()[0]:.4f}")

# Evaluation
test_outputs = model(X_test).numpy()
y_pred = np.argmax(test_outputs, axis=1)
y_true = np.argmax(y_test_np, axis=1)
accuracy = (y_pred == y_true).sum() / y_true.shape[0]
print(f"Test Accuracy: {accuracy:.4f}")
