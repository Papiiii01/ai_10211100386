import torch

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple tensor
x = torch.rand(5, 3)
print("\nRandom tensor:")
print(x)

# Basic operation
y = x + 1
print("\nTensor after adding 1:")
print(y) 