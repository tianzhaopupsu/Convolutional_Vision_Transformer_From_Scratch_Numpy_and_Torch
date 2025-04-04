import numpy as np

class Convit:
    def __init__(self,data):
        
        self.data=np.array(data)
        # Hyperparameters you migh need to adjust
        self.patch_size=4   #### the patch number for attention
        self.embed_dim=6    #### what is the dimension you want for each token in attention
        self.filter_size=3  #### kernal size of the convolution, we use same size for both of the layers
        self.stride=1       #### stride size
        self.padding=1      #### extra 0 padding
        self.pool_size=2    #### the pool size for maxpooling
        self.D_hidden = 2*self.embed_dim  # FFN hidden layer size (typically 2-4x D)
        self.C = 3  # Number of classes
        self.learning_rate = 0.01  # Learning rate
        self.num_heads = 2   ### number of attention heads, for simplicity, it is 2 here    
        return

    def pre_training(self):
        # initialize some parameters for the future processing
        self.batch_size=self.data.shape[0]
        self.height=self.data.shape[1]
        self.width=self.data.shape[2]
        self.num_patches = (self.height//4 // self.patch_size) * (self.width//4 // self.patch_size)  # Total patches
        
        # Convolutional Layer
        # Initialize CNN Kernel, the initial guess is always small random numbers
        # Classic 2 filters design


    def start_training(self):
        self.pre_training()

        N, T, D = self.batch_size, self.num_patches, self.embed_dim  # Batch size, Tokens (Patches), Feature dimension
        head_dim = D // self.num_heads  # Dimension per head
        conv_kernel1 = np.random.randn(self.filter_size, self.filter_size)
        conv_kernel2 = np.random.randn(self.filter_size, self.filter_size)
        W_embed = np.random.randn(self.patch_size * self.patch_size, self.embed_dim)
        pos_embed = np.random.randn(1, self.num_patches, self.embed_dim)  # Learnable positional embedding

        # Initialize Attention Weights
        W_q = np.random.randn(D, D)  # Query weight matrix
        W_k = np.random.randn(D, D)  # Key weight matrix
        W_v = np.random.randn(D, D)  # Value weight matrix

        # Initialize FFN Weights (Linear -> ReLU -> Linear)
        W1 = np.random.randn(D, self.D_hidden)  # First linear layer
        B1 = np.random.randn(self.D_hidden)
        W2 = np.random.randn(self.D_hidden, D)  # Second linear layer
        B2 = np.random.randn(D)
        W_o=np.random.rand(D,D)
        # Initialize Final Linear Weights
        W_linear = np.random.randn(D, self.C)  # Final classification layer
        B_linear = np.random.randn(self.C)  # Bias for the linear layer



        #######In general, you want to loop this part #####################
        #######to train the model based on your file strucutre#######################################
        #########################################################################
        #########################################################################
        ###################### The convolution section, consisted by 2-filter setup######
        X_conv1 = np.array([self.conv2d(img, conv_kernel1) for img in self.data])  # Apply to batch
        # ReLU Activation
        X_relu1 = np.maximum(0, X_conv1)
        # Max Pooling (pool_size*pool_size)
        X_pooled1 = np.array([self.max_pooling(img, self.pool_size) for img in X_relu1])
        ###Second convolutional layer
        X_conv2 = np.array([self.conv2d(img, conv_kernel2) for img in X_pooled1])  # Apply to batch
        # ReLU Activation
        X_relu2 = np.maximum(0, X_conv2)
        ###Second Max pooling
        X_pooled2 = np.array([self.max_pooling(img, self.pool_size) for img in X_relu2])
        # Convert Image to Patches
        X_patches = X_pooled2.reshape(self.batch_size, self.num_patches, self.patch_size * self.patch_size)
        # Linear Projection to Embedding Dimension
        X_embedded = np.matmul(X_patches, W_embed)  # (batch_size, num_patches, embed_dim)
        # Positional Embedding
        X_input_MHA = X_embedded + pos_embed  # Final input before MHA
        print("Input to Multi-Head Attention Shape:", X_input_MHA.shape)
        

        #################### Multi-Head Self-Attention################
        # Linear projections for Q, K, V
        Q = np.matmul(X_input_MHA, W_q).reshape(N, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
        K = np.matmul(X_input_MHA, W_k).reshape(N, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)
        V = np.matmul(X_input_MHA, W_v).reshape(N, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

        ##################### Scaled Dot-Product Attention
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
        y_true = np.zeros((N, self.C))
        y_true[np.arange(N), np.random.randint(0, self.C, size=N)] = 1  # Random ground truth
        # Cross-Entropy Loss
        loss = -np.sum(y_true * np.log(softmax_output + 1e-9)) / N  # Avoid log(0)
        
        print("The loss of current iteration:", loss)
        ######################  Backpropagation starts from here, Loss gradient will be calculated
        # Gradient of Loss w.r.t Softmax Output
        dL_dSoftmax = softmax_output - y_true  # (N, C)

        # Gradient w.r.t Logits
        dL_dLogits = dL_dSoftmax / N  # (N, C)

        # Gradient w.r.t Final Linear Weights and Bias
        dL_dW_linear = np.dot(gap_output.T, dL_dLogits)  # (D, C)
        dL_dB_linear = np.sum(dL_dLogits, axis=0)  # (C,)

        # Update Final Linear Weights and Bias
        W_linear -= self.learning_rate * dL_dW_linear
        B_linear -= self.learning_rate * dL_dB_linear

        # Gradient w.r.t GAP output
        dL_dGAP = np.dot(dL_dLogits, W_linear.T)  # (N, D)

        # Gradient w.r.t Feature Map before GAP
        dL_dFeatureMap = np.repeat(dL_dGAP[:, np.newaxis, :], T, axis=1) / T  # (N, T, D)

        dL_dFeatureMap_after_norm =dL_dFeatureMap * (1 / np.std(X_residual2, axis=-1, keepdims=True))  # Norm backpropagation
        dL_dFeatureMap_after_norm -= dL_dFeatureMap_after_norm.mean(axis=1, keepdims=True)  # Residual subtraction

        # Backprop through FFN
        dL_dFFN2 = dL_dFeatureMap_after_norm  # Gradient from GAP
        dL_dW2 = np.matmul(X_ffn_relu.reshape(N * T, self.D_hidden).T, dL_dFFN2.reshape(N * T, D))  # (D_hidden, D)  # (D_hidden, D)
        dL_dB2 = np.sum(dL_dFFN2.reshape(N * T, D), axis=0)
        # Update FFN Weights and Biases
        W2 -= self.learning_rate * dL_dW2
        B2 -= self.learning_rate * dL_dB2

        # Backprop through ReLU
        dL_dReLU = np.dot(dL_dFFN2, W2.T)
        dL_dReLU[X_ffn1 <= 0] = 0  # Zero out gradient where ReLU was inactive

        # Backprop through first FFN layer
        dL_dW1 = np.matmul(X_norm.reshape(N * T, D).T, dL_dReLU.reshape(N * T, self.D_hidden))  # (D, D_hidden)  # (D, D_hidden)
        dL_dB1 = np.sum(dL_dReLU.reshape(N * T, self.D_hidden), axis=0)

        # Update First FFN Layer Weights and Biases
        W1 -= self.learning_rate * dL_dW1
        B1 -= self.learning_rate * dL_dB1

        # Backprop through Add & Norm
        dL_dFeatureMap_outFFN=np.dot(dL_dReLU, W1.T)
        dL_dFeatureMap_outFFN = dL_dFeatureMap_outFFN * (1 / np.std(X_residual, axis=-1, keepdims=True))


        dL_dAttention_output = dL_dFeatureMap_outFFN.reshape(N, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)  # (N, H, T, d_h)

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
        W_v -= self.learning_rate * dL_dW_v
        W_k -= self.learning_rate * dL_dW_k
        W_q -= self.learning_rate * dL_dW_q
        # Backprop through Add & Norm

        # Backprop through Linear Projection and Learnable Position Embedding
        dL_dLinearProj = np.dot(dL_dFeatureMap_outFFN, W_embed.T)
        dL_dPositionEmbedding = np.sum(dL_dFeatureMap_outFFN, axis=0, keepdims=True)
        W_embed-=self.learning_rate * np.matmul(X_embedded.reshape(N * T, D).T, dL_dLinearProj.reshape(N * T, self.patch_size * self.patch_size)).T
        pos_embed -= self.learning_rate * dL_dPositionEmbedding

        # Backprop through Max Pooling
        #### this will convert the tokenized image back to the original dimension in convolutional module
        dL_dImage=self.reconstruct_image_gradient(dL_dLinearProj, (self.batch_size, self.patch_size*2, self.patch_size*2), self.patch_size)
        dL_dMaxPool2 = dL_dImage.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2)
        dL_dReLU2 = dL_dMaxPool2 * (X_conv2 > 0)  # Backprop through second ReLU
        flipped_filters = np.flip(conv_kernel2, axis=(0, 1))
        dL_dConv2=self.reconstruct_filter(dL_dReLU2,X_pooled2)
        # Update filter weights
        conv_kernel2 -= self.learning_rate * np.mean(dL_dConv2,axis=0)
        # Flip the kernel for proper backpropagation
        W_flipped = np.flip(conv_kernel2)
        
        dL_dOriginal=self.reconstruct_convolimage(dL_dReLU2,X_pooled2,W_flipped)

        # Backprop through first max pooling
        dL_dMaxPool1 = dL_dOriginal.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2)
        dL_dReLU1 = dL_dMaxPool1 * (X_conv1 > 0)  # Backprop through first ReLU
        dL_dConv1=self.reconstruct_filter(dL_dReLU1,X_pooled1)
        # Update first convolutional layer weights
        conv_kernel1 -= self.learning_rate * np.mean(dL_dConv1,axis=0)
        print("All parameters have been updated, going to next round")
    ################################ below are the functions to support the above training
    ########################################################################################
    def conv2d(self,img, kernel):
        h, w = img.shape
        k = kernel.shape[0]
        pad = (k - 1) // 2
        img_padded = np.pad(img, pad, mode='constant', constant_values=0)
        output = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(img_padded[i:i+k, j:j+k] * kernel)
        return output
    def max_pooling(self,img, pool_size):
        h, w = img.shape
        new_h, new_w = h // pool_size, w // pool_size
        output = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                output[i, j] = np.max(img[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
        return output

    def reconstruct_image_gradient(self,dL_dPatches, image_shape, patch_size):
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
    def reconstruct_filter(self,pre_loss,pre_input):
        dL_dConv2 = np.zeros((self.batch_size,self.filter_size,self.filter_size))
        # Compute gradient of filter weights
        for b in range(pre_loss.shape[0]):
            dsample=np.array(pre_loss[0,:,:])
            sample=np.array(pre_input[0,:,:])
            h, w = sample.shape
            k = pre_loss.shape[0]
            pad = (k - 1) // 2
            img_padded = np.pad(sample, pad, mode='constant', constant_values=0)
            for i in range(h):
                for j in range(w):
                    dL_dConv2 += dsample[i, j] *np.sum(img_padded[i:i+k, j:j+k])
        return dL_dConv2
    def reconstruct_convolimage(self,pre_loss,pre_input,W_flipped):
        dL_dX=np.zeros((self.batch_size,18,18)) 
        for b in range(pre_loss.shape[0]):
            dsample=np.array(pre_loss[0,:,:])
            sample=np.array(pre_input[0,:,:])
            h, w = sample.shape
            k = W_flipped.shape[0]
            pad = (k - 1) // 2
            img_padded = np.pad(sample, pad, mode='constant', constant_values=0)
            
            for i in range(h):
                for j in range(w):
                    dL_dX[b, i:i+k, j:j+k] += pre_loss[b, i, j] * W_flipped 
        return dL_dX[:, pad:-pad, pad:-pad]


X = np.random.randn(2, 32, 32)
ans=Convit(X)
ans.start_training()
