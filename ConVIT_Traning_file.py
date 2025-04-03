import numpy as np


# Hyperparameters
batch_size = 2
height = 32
width = 32
patch_size = 4  # Patch size for tokenization
embed_dim = 4  # Embedding dimension
num_patches = (height//4 // patch_size) * (width//4 // patch_size)  # Total patches

# Input Data (batch, height, width)
X = np.random.randn(batch_size, height, width)

# Convolutional Layer
filter_size = 3  # Kernel size
stride = 1  # Stride for convolution
padding = 1  # Padding

# Initialize CNN Kernel
conv_kernel1 = np.random.randn(filter_size, filter_size)
conv_kernel2 = np.random.randn(filter_size, filter_size)
W_embed = np.random.randn(patch_size * patch_size, embed_dim)
pos_embed = np.random.randn(1, num_patches, embed_dim)  # Learnable positional embedding

# Apply Convolution (Valid Padding)
def conv2d(img, kernel):
    h, w = img.shape
    k = kernel.shape[0]
    pad = (k - 1) // 2
    img_padded = np.pad(img, pad, mode='constant', constant_values=0)
    output = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(img_padded[i:i+k, j:j+k] * kernel)
    return output

X_conv1 = np.array([conv2d(img, conv_kernel1) for img in X])  # Apply to batch

# ReLU Activation
X_relu1 = np.maximum(0, X_conv1)



# Max Pooling (2x2)
pool_size = 2
def max_pooling(img, pool_size):
    h, w = img.shape
    new_h, new_w = h // pool_size, w // pool_size
    output = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            output[i, j] = np.max(img[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    return output

X_pooled1 = np.array([max_pooling(img, pool_size) for img in X_relu1])

###Second convolutional layer
X_conv2 = np.array([conv2d(img, conv_kernel2) for img in X_pooled1])  # Apply to batch

# ReLU Activation
X_relu2 = np.maximum(0, X_conv2)
###Second Max pooling
X_pooled2 = np.array([max_pooling(img, pool_size) for img in X_relu2])

# Convert Image to Patches
X_patches = X_pooled2.reshape(batch_size, num_patches, patch_size * patch_size)
print(X_patches.shape)
# Linear Projection to Embedding Dimension

X_embedded = np.matmul(X_patches, W_embed)  # (batch_size, num_patches, embed_dim)
print(X_embedded.shape)
# Positional Embedding
X_input_MHA = X_embedded + pos_embed  # Final input before MHA

print("Input to Multi-Head Attention Shape:", X_input_MHA.shape)


# Hyperparameters
N, T, D = batch_size, num_patches, embed_dim  # Batch size, Tokens (Patches), Feature dimension
D_hidden = 2*D  # FFN hidden layer size (typically 2-4x D)
C = 3  # Number of classes
learning_rate = 0.01  # Learning rate
num_heads = 2
head_dim = D // num_heads  # Dimension per head

# Initialize Attention Weights
W_q = np.random.randn(D, D)  # Query weight matrix
W_k = np.random.randn(D, D)  # Key weight matrix
W_v = np.random.randn(D, D)  # Value weight matrix

# Initialize FFN Weights (Linear -> ReLU -> Linear)
W1 = np.random.randn(D, D_hidden)  # First linear layer
B1 = np.random.randn(D_hidden)
W2 = np.random.randn(D_hidden, D)  # Second linear layer
B2 = np.random.randn(D)
W_o=np.random.rand(D,D)
# Initialize Final Linear Weights
W_linear = np.random.randn(D, C)  # Final classification layer
B_linear = np.random.randn(C)  # Bias for the linear layer

# Feature Map (Input to Attention)
  # (Batch, Tokens, Feature dim)

# Multi-Head Self-Attention


# Linear projections for Q, K, V
Q = np.matmul(X_input_MHA, W_q).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
K = np.matmul(X_input_MHA, W_k).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
V = np.matmul(X_input_MHA, W_v).reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

# Scaled Dot-Product Attention
attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)  # (N, H, T, T)
norm_scores=np.exp(attention_scores-np.max(attention_scores, axis=-1, keepdims=True))
attention_weights = norm_scores / np.sum(norm_scores, axis=-1, keepdims=True)  # Softmax
attention_output = np.matmul(attention_weights, V)  # (N, H, T, d_h)

# Merge heads back
attention_output = attention_output.transpose(0, 2, 1, 3).reshape(N, T, D)  # (N, T, D)

# Output projection
attention_output = np.matmul(attention_output, W_o)  # (N, T, D)

# Add & Norm
X_residual = X_input_MHA + attention_output  # Skip connection
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

dL_dFeatureMap_after_norm =dL_dFeatureMap * (1 / np.std(X_residual2, axis=-1, keepdims=True))  # Norm backpropagation
dL_dFeatureMap_after_norm -= dL_dFeatureMap_after_norm.mean(axis=1, keepdims=True)  # Residual subtraction

# Backprop through FFN
dL_dFFN2 = dL_dFeatureMap_after_norm  # Gradient from GAP
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
dL_dFeatureMap_outFFN=np.dot(dL_dReLU, W1.T)
dL_dFeatureMap_outFFN = dL_dFeatureMap_outFFN * (1 / np.std(X_residual, axis=-1, keepdims=True))


dL_dAttention_output = dL_dFeatureMap_outFFN.reshape(N, T, num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

dL_dV = np.matmul(attention_weights.transpose(0, 1, 3, 2), dL_dAttention_output)  # (N, H, T, d_h)
dL_dAttention_weights = np.matmul(dL_dAttention_output, V.transpose(0, 1, 3, 2))  # (N, H, T, T)
dL_dAttention_scores = dL_dAttention_weights * attention_weights * (1 - attention_weights)  # Softmax derivative

dL_dK = np.matmul(dL_dAttention_scores, Q)  # (N, H, T, d_h)
dL_dQ = np.matmul(dL_dAttention_scores.transpose(0, 1, 3, 2), K)  # (N, H, T, d_h)

# Compute gradients w.r.t. weight matrices
dL_dW_v = np.matmul(X_input_MHA.reshape(N * T, D).T, dL_dV.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)
dL_dW_k = np.matmul(X_input_MHA.reshape(N * T, D).T, dL_dK.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)
dL_dW_q = np.matmul(X_input_MHA.reshape(N * T, D).T, dL_dQ.transpose(0, 2, 1, 3).reshape(N * T, D))  # (D, D)

# Update Multi-Head Attention Weights
W_v -= learning_rate * dL_dW_v
W_k -= learning_rate * dL_dW_k
W_q -= learning_rate * dL_dW_q


# Backprop through Add & Norm

# Backprop through Linear Projection and Learnable Position Embedding
dL_dLinearProj = np.dot(dL_dFeatureMap_outFFN, W_embed.T)
dL_dPositionEmbedding = np.sum(dL_dFeatureMap_outFFN, axis=0, keepdims=True)
W_embed-=learning_rate * np.matmul(X_embedded.reshape(N * T, D).T, dL_dLinearProj.reshape(N * T, patch_size * patch_size)).T

pos_embed -= learning_rate * dL_dPositionEmbedding

# Backprop through Max Pooling
def reconstruct_image_gradient(dL_dPatches, image_shape, patch_size):
    batch_size, H, W = image_shape
    P = patch_size
    
    # Compute number of patches along height and width
    num_patches_h = H // P
    num_patches_w = W // P
    
    # Reshape from (batch, num_patches, P*P) → (batch, num_patches, P, P)
    dL_dPatches = dL_dPatches.reshape(batch_size, num_patches_h * num_patches_w, P, P)

    # Create empty gradient array
    dL_dImage = np.zeros((batch_size, H, W))

    # Reconstruct image from patches
    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            dL_dImage[:, i * P:(i + 1) * P, j * P:(j + 1) * P] = dL_dPatches[:, patch_idx]
            patch_idx += 1

    return dL_dImage

dL_dImage=reconstruct_image_gradient(dL_dLinearProj, (batch_size, patch_size*2, patch_size*2), patch_size)

dL_dMaxPool2 = dL_dImage.repeat(pool_size, axis=1).repeat(pool_size, axis=2)

dL_dReLU2 = dL_dMaxPool2 * (X_conv2 > 0)  # Backprop through second ReLU

dL_dConv2 = np.zeros((batch_size,filter_size,filter_size))
k = conv_kernel2.shape[0]
pad = (k - 1) // 2
dL_dX=np.zeros((batch_size,18,18)) 
flipped_filters = np.flip(conv_kernel2, axis=(0, 1))

    # Compute gradient of filter weights
for b in range(dL_dReLU2.shape[0]):
    dsample=np.array(dL_dReLU2[0,:,:])
    sample=np.array(X_pooled2[0,:,:])
    h, w = sample.shape
    k = conv_kernel2.shape[0]
    pad = (k - 1) // 2
    img_padded = np.pad(sample, pad, mode='constant', constant_values=0)
      
    for i in range(h):
        for j in range(w):

            dL_dConv2 += dsample[i, j] *np.sum(img_padded[i:i+k, j:j+k])


# Update filter weights
conv_kernel2 -= learning_rate * np.mean(dL_dConv2,axis=0)


# Flip the kernel for proper backpropagation
W_flipped = np.flip(conv_kernel2)

for b in range(dL_dReLU2.shape[0]):
    dsample=np.array(dL_dReLU2[0,:,:])
    sample=np.array(X_pooled2[0,:,:])
    h, w = sample.shape
    k = conv_kernel2.shape[0]
    pad = (k - 1) // 2
    img_padded = np.pad(sample, pad, mode='constant', constant_values=0)
      
    for i in range(h):
        for j in range(w):

            dL_dX[b, i:i+k, j:j+k] += dL_dReLU2[b, i, j] * W_flipped 
dL_dOriginal = dL_dX[:, pad:-pad, pad:-pad]

# Backprop through first max pooling
dL_dMaxPool1 = dL_dOriginal.repeat(pool_size, axis=1).repeat(pool_size, axis=2)
dL_dReLU1 = dL_dMaxPool1 * (X_conv1 > 0)  # Backprop through first ReLU

dL_dConv1 = np.zeros((batch_size,filter_size,filter_size))
for b in range(dL_dReLU1.shape[0]):
    dsample=np.array(dL_dReLU1[0,:,:])
    sample=np.array(X_pooled1[0,:,:])
    h, w = sample.shape
    k = conv_kernel1.shape[0]
    pad = (k - 1) // 2
    img_padded = np.pad(sample, pad, mode='constant', constant_values=0)
      
    for i in range(h):
        for j in range(w):

            dL_dConv1 += dsample[i, j] *np.sum(img_padded[i:i+k, j:j+k])

print(dL_dConv1.shape)
# Update first convolutional layer weights
conv_kernel1 -= learning_rate * np.mean(dL_dConv1,axis=0)


