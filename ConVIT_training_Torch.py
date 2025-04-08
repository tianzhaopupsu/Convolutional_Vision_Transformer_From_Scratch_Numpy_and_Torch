import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ConViT(nn.Module):
    def __init__(self, img_size=16, patch_size=4, embed_dim=64, heads=4, hidden_dim=128, num_classes=3):
        super().__init__()
        self.conv1 = ConvBlock()
        self.conv2 = ConvBlock()
        self.embed = PatchEmbedding(img_size//4, patch_size, embed_dim)
        self.transformer = TransformerBlock(embed_dim, heads, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, self.patch_size * self.patch_size)
        x = self.proj(x)
        x = x + self.pos_embed
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, self.head_dim).transpose(1, 2) for t in qkv]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

import torch.optim as optim
# Assuming model is your ConVIT model
model = ConViT().to('cpu')  # device = 'cuda' or 'cpu'
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
num_epochs=5

images = torch.randn(10, 1, 16, 16)  # Example data: 64 images of size 1x16x16
labels = torch.randint(0, 3, (10,))  # Example labels: 64 labels between 0 and 9
# Create a TensorDataset
dataset = TensorDataset(images, labels)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True) 

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        # images, labels = images.to(device), labels.to(device) # If using GPU, uncomment this line

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
