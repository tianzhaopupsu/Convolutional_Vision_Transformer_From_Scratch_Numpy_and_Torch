# Convolutional_Vision_Transformer_From_Scratch_Numpy_and_Torch

This is to share the Python file to build a Convolutional Vision Transformer from scratch. The purpose of the Numpy-only is to show the important steps that might not be seen by using Pytorch or other packages.

Logic_Maths_Numpy_only folder, there are three files here:

ConVIT_logic_flow.py gives the very nature flow of the whole module without heavily using functions. It might be easy for first time learner.

ConVIT_training_file.py this is to train your model, but some of hyperparameters need to be adjust first before the training, they all marked.

ConVIT_work_AI.py basically, after the training, with the updated weights, this file can run well to predict/descirbe the image.

All the files can run directly, some example random matrix was fabracated as the input data to make sure the code can run, at least.

The Torch verion is in the main root, ConVIT_training_Torch.py is for training purpose, and ConVIT_Torch_wor_AI.py is the one to run after training.

The ConVIT is built by the following architecture: 

1. Two Convolutional Layers: The model starts with two convolutional layers that extract local features from the input image. These layers use small filters to capture fine-grained details from the image.

2. Pooling Layers: After each convolutional layer, max-pooling layers are applied to reduce the spatial dimensions of the image while retaining important features. This helps make the model more computationally efficient.

3. Patch Embedding: The image is divided into smaller patches, and each patch is flattened and projected into a higher-dimensional embedding space. This helps transform the image into a format suitable for the transformer module.

4. Positional Embedding: Since the transformer model doesn't inherently understand the spatial relationships between patches, learnable positional embeddings are added to each patch embedding to preserve the relative positions of the patches.

5. Multi-Head Self-Attention: The core of the transformer is the multi-head self-attention mechanism. It allows the model to capture relationships between all patches in the image, regardless of their position. This helps the model focus on important global features across the image.

6. Feed-Forward Network (FFN): After the attention mechanism, the output is passed through a feed-forward network, which consists of two linear layers with ReLU activation in between. This helps transform the features into a more useful representation.

7. Global Average Pooling (GAP): The output from the FFN is pooled across all patches, summarizing the information into a single vector that represents the entire image.

8. Final Classification Layer: The pooled vector is passed through a final linear layer to output the predicted class probabilities.

Have fun.
