from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils
import os

# Output directory for digit images
os.makedirs("data/digits", exist_ok=True)

# Load MNIST
mnist = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(mnist, batch_size=1, shuffle=True)

# Save first 1000 digits as PNG files
for idx, (img, label) in enumerate(loader):
    if idx >= 1000:
        break
    torchvision.utils.save_image(img, f"data/digits/{label.item()}_{idx}.png")
