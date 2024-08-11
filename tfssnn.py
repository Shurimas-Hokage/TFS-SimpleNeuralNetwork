import numpy as np

def relu(x, deriv=False):
    if deriv:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

# input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output data
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# synapses
syn0 = np.random.randn(3, 4)
syn1 = np.random.randn(4, 1)

# hyperparameters
learning_rate = 0.1
dropout_rate = 0.2
batch_size = 2
num_epochs = 50000
lr_decay = 0.9
lr_decay_step = 10000
best_val_loss = float('inf')
patience = 5000
early_stop_counter = 0

# training loop
for epoch in range(num_epochs):
    # shuffle the training data
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # mini-batch training
    for i in range(0, X.shape[0], batch_size):
        # Forward propagation
        l0 = X[i:i+batch_size]
        l1 = relu(np.dot(l0, syn0))
        l1_dropout = np.where(np.random.rand(*l1.shape) < dropout_rate, 0, l1) / (1 - dropout_rate)
        l2 = relu(np.dot(l1_dropout, syn1))

        # Backpropagation
        l2_error = y[i:i+batch_size] - l2
        l2_delta = l2_error * relu(l2, deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * relu(l1, deriv=True)

        # Weight updates
        syn1 += learning_rate * l1_dropout.T.dot(l2_delta)
        syn0 += learning_rate * l0.T.dot(l1_delta)

    # Learning rate scheduling
    if (epoch+1) % lr_decay_step == 0:
        learning_rate *= lr_decay

# Validation
    if (epoch+1) % 1000 == 0:
        l1_val = relu(np.dot(X, syn0))
        l2_val = relu(np.dot(l1_val, syn1))
        val_loss = np.mean(np.abs(y - l2_val))
        print("Epoch:", epoch+1, "Validation Loss:", val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping...")
                break

# Final evaluation
l1_final = relu(np.dot(X, syn0))
l2_final = relu(np.dot(l1_final, syn1))
print("Output after training")
print(l2_final)