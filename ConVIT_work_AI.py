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


    def tell_the_truth(self):
        self.pre_training()

        N, T, D = self.batch_size, self.num_patches, self.embed_dim  # Batch size, Tokens (Patches), Feature dimension
        head_dim = D // self.num_heads  # Dimension per head



######################### all the weights need to be replaced by the trained value
        ############################################################
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
#################################################################################


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
        print(softmax_output)
        #################################################################################
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


X = np.random.randn(1, 32, 32)
ans=Convit(X)
ans.tell_the_truth()
