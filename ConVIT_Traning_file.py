import numpy as np

# Hyperparameters
N, T, D = 2, 3, 4  # Batch size, Tokens (Patches), Feature dimension
D_hidden = 8  # FFN hidden layer size (typically 2-4x D)
C = 3  # Number of classes
learning_rate = 0.01  # Learning rate

# Initialize Attention Weights
W_q = np.random.randn(D, D)  # Query weight matrix
W_k = np.random.randn(D, D)  # Key weight matrix
W_v = np.random.randn(D, D)  # Value weight matrix

# Initialize FFN Weights (Linear -> ReLU -> Linear)
W1 = np.random.randn(D, D_hidden)  # First linear layer
B1 = np.random.randn(D_hidden)
W2 = np.random.randn(D_hidden, D)  # Second linear layer
B2 = np.random.randn(D)
W_o=np.random.rand(4,4)
# Initialize Final Linear Weights
W_linear = np.random.randn(D, C)  # Final classification layer
B_linear = np.random.randn(C)  # Bias for the linear layer

# Feature Map (Input to Attention)
X=np.array([[[1,2,3,-4],[5,6,-7,8],[5,6,-7,8]],[[1,2,3,4],[5,6,7,8],[5,6,-7,8]]])
  # (Batch, Tokens, Feature dim)

# Multi-Head Self-Attention
num_heads = 2
head_dim = D // num_heads  # Dimension per head

# Linear projections for Q, K, V
Q = np.matmul(X, W_q).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
K = np.matmul(X, W_k).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
V = np.matmul(X, W_v).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

# Scaled Dot-Product Attention
attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)  # (N, H, T, T)
attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)  # Softmax
attention_output = np.matmul(attention_weights, V)  # (N, H, T, d_h)

# Merge heads back
attention_output = attention_output.transpose(0, 2, 1, 3).reshape(N, T, D)  # (N, T, D)

# Output projection
attention_output = np.matmul(attention_output, W_o)  # (N, T, D)

# Add & Norm
X_residual = X + attention_output  # Skip connection
X_norm = (X_residual - np.mean(X_residual, axis=-1, keepdims=True)) / (np.std(X_residual, axis=-1, keepdims=True) + 1e-6)

# FFN: Linear -> ReLU -> Linear
X_ffn1 = np.dot(X_norm, W1) + B1  # First linear transformation
X_ffn_relu = np.maximum(0, X_ffn1)  # ReLU activation
X_ffn2 = np.dot(X_ffn_relu, W2) + B2  # Second linear transformation

# Add & Norm
X_residual2 = X_norm + X_ffn2  # Skip connection 2
X_norm2 = (X_residual2 - np.mean(X_residual2, axis=-1, keepdims=True)) / (np.std(X_residual2, axis=-1, keepdims=True) + 1e-6)

# Global Average Pooling (GAP)
gap_output = np.mean(X_norm2, axis=1)  # (N, D)

# Linear Transformation
logits = np.dot(gap_output, W_linear) + B_linear  # (N, C)

# Softmax
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability trick
softmax_output = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (N, C)

# Ground truth labels (one-hot)
y_true = np.zeros((N, C))
y_true[np.arange(N), np.random.randint(0, C, size=N)] = 1  # Random ground truth

# Cross-Entropy Loss
loss = -np.sum(y_true * np.log(softmax_output + 1e-9)) / N  # Avoid log(0)



# Gradient of Loss w.r.t Softmax Output
dL_dSoftmax = softmax_output - y_true  # (N, C)

# Gradient w.r.t Logits
dL_dLogits = dL_dSoftmax / N  # (N, C)

# Gradient w.r.t Final Linear Weights and Bias
dL_dW_linear = np.dot(gap_output.T, dL_dLogits)  # (D, C)
dL_dB_linear = np.sum(dL_dLogits, axis=0)  # (C,)

# Update Final Linear Weights and Bias
W_linear -= learning_rate * dL_dW_linear
B_linear -= learning_rate * dL_dB_linear

# Gradient w.r.t GAP output
dL_dGAP = np.dot(dL_dLogits, W_linear.T)  # (N, D)

# Gradient w.r.t Feature Map before GAP
dL_dFeatureMap = np.repeat(dL_dGAP[:, np.newaxis, :], T, axis=1) / T  # (N, T, D)

# Backprop through FFN
dL_dFFN2 = dL_dFeatureMap  # Gradient from GAP
dL_dW2 = np.matmul(X_ffn_relu.reshape(N * T, D_hidden).T, dL_dFFN2.reshape(N * T, D))  # (D_hidden, D)  # (D_hidden, D)
dL_dB2 = np.sum(dL_dFFN2.reshape(N * T, D), axis=0)
# Update FFN Weights and Biases
W2 -= learning_rate * dL_dW2
B2 -= learning_rate * dL_dB2

# Backprop through ReLU
dL_dReLU = np.dot(dL_dFFN2, W2.T)
dL_dReLU[X_ffn1 <= 0] = 0  # Zero out gradient where ReLU was inactive

# Backprop through first FFN layer
dL_dW1 = np.matmul(X_norm.reshape(N * T, D).T, dL_dReLU.reshape(N * T, D_hidden))  # (D, D_hidden)  # (D, D_hidden)
dL_dB1 = np.sum(dL_dReLU.reshape(N * T, D_hidden), axis=0)

# Update First FFN Layer Weights and Biases
W1 -= learning_rate * dL_dW1
B1 -= learning_rate * dL_dB1

# Backprop through Add & Norm
dL_dFeatureMap_before_Norm = dL_dReLU * (1 / np.std(X_residual, axis=-1, keepdims=True))


dL_dAttention_output = dL_dFeatureMap.reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

dL_dV = np.matmul(attention_weights.transpose(0, 1, 3, 2), dL_dAttention_output)  # (N, H, T, d_h)
dL_dAttention_weights = np.matmul(dL_dAttention_output, V.transpose(0, 1, 3, 2))  # (N, H, T, T)
dL_dAttention_scores = dL_dAttention_weights * attention_weights * (1 - attention_weights)  # Softmax derivative

dL_dK = np.matmul(dL_dAttention_scores, Q)  # (N, H, T, d_h)
dL_dQ = np.matmul(dL_dAttention_scores.transpose(0, 1, 3, 2), K)  # (N, H, T, d_h)
print(dL_dV.transpose(0, 2, 1, 3).reshape(N, T, D).shape,X.T.shape)
# Compute gradients w.r.t. weight matrices
dL_dW_v = np.matmul(X.reshape(N * T, D).T, dL_dV.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)
dL_dW_k = np.matmul(X.reshape(N * T, D).T, dL_dK.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)
dL_dW_q = np.matmul(X.reshape(N * T, D).T, dL_dQ.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)

# Update Multi-Head Attention Weights
W_v -= learning_rate * dL_dW_v
W_k -= learning_rate * dL_dW_k
W_q -= learning_rate * dL_dW_q
