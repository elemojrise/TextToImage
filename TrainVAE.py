import os
import csv
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ==========================
# 1. DATASET DEFINITION
# ==========================
class ShapeDataset(Dataset):
    """
    A PyTorch dataset that reads images and (optionally) text descriptions
    from a CSV file. Here we'll only use the images for VAE training.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Row structure: "image_path", "text_description"
                img_path = row["image_path"]
                self.image_paths.append(img_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

# ==========================
# 2. VAE MODEL DEFINITION
# ==========================
class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        """
        Very minimal VAE for 64x64 images, downsampling to an 8x8 feature map.
        latent_dim is the number of channels in that final feature map 
        (we flatten or keep it as a 2D).
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # ENCODER
        # input: (3, 64, 64)
        # We want to eventually get to (latent_dim, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# -> (128, 8, 8)
            nn.ReLU(True),
        )
        
        # We'll map from (128, 8, 8) -> two different "heads" for mu and log_var
        # Flatten the (128, 8, 8) = 8192 features
        self.flatten_size = 128 * 8 * 8
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim * 8 * 8)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim * 8 * 8)
        
        # DECODER
        # We'll decode from (latent_dim, 8, 8) back to (3, 64, 64)
        self.decoder_input = nn.Linear(latent_dim * 8 * 8, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # -> (3, 64, 64)
            nn.Sigmoid()  # output in [0,1]
        )
        
    def encode(self, x):
        """
        Returns the latent distribution parameters (mu, log_var).
        x shape: (batch, 3, 64, 64)
        """
        h = self.encoder(x)  # -> (batch, 128, 8, 8)
        h = h.view(h.size(0), -1)  # flatten
        mu = self.fc_mu(h)        # (batch, latent_dim*8*8)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * eps
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent z to reconstruct the image.
        z shape: (batch, latent_dim*8*8)
        """
        h = self.decoder_input(z)  # (batch, 8192)
        h = h.view(h.size(0), 128, 8, 8)  # reshape to (batch, 128, 8, 8)
        x_recon = self.decoder(h)  # -> (batch, 3, 64, 64)
        return x_recon
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    """
    Standard VAE loss = reconstruction loss + KLD
    - recon_x: reconstructed image
    - x: original image
    - mu, log_var: VAE latent distribution parameters
    """
    # Reconstruction loss (MSE or BCE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # KLD = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kld

# ==========================
# 3. TRAINING LOOP
# ==========================
def train_vae(
    data_loader, 
    model, 
    optimizer, 
    device='cpu', 
    epochs=20,
    log_interval=100
):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, images in enumerate(data_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            recon_images, mu, log_var = model(images)
            loss = vae_loss(recon_images, images, mu, log_var)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {avg_loss:.2f}")
                total_loss = 0
        
        # End of epoch, you can sample or visualize reconstructions
        sample_visualization(model, images, recon_images, epoch, device=device)

def sample_visualization(model, real_imgs, recon_imgs, epoch, device='cpu'):
    """
    Plots real images vs. reconstructed images for a quick check.
    """
    model.eval()
    # Plot first 6 examples
    num_show = min(6, real_imgs.size(0))
    
    real_imgs = real_imgs.detach().cpu().numpy()
    recon_imgs = recon_imgs.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, num_show, figsize=(num_show*2, 4))
    for i in range(num_show):
        # Real
        axes[0, i].imshow(np.transpose(real_imgs[i], (1, 2, 0)))
        axes[0, i].axis('off')
        # Recon
        axes[1, i].imshow(np.transpose(recon_imgs[i], (1, 2, 0)))
        axes[1, i].axis('off')
        
    plt.suptitle(f"Epoch {epoch+1} Reconstruction")
    plt.show()
    model.train()


if __name__ == "__main__":
    # ==========================
    # 1. Setup
    # ==========================
    DATA_FOLDER = "C:\Data"        # folder containing images
    CSV_FILE = os.path.join(DATA_FOLDER, "trainingData.csv")
    BATCH_SIZE = 16
    EPOCHS = 15  # Increase as needed (e.g., 50-100)
    LR = 1e-3    # learning rate
    
    # ==========================
    # 2. Create dataset & loader
    # ==========================
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    dataset = ShapeDataset(
        csv_file=CSV_FILE,
        root_dir=DATA_FOLDER,
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # ==========================
    # 3. Model, optimizer
    # ==========================
    # Let's set latent_dim=16 or 32 for minimal complexity
    vae_model = VAE(latent_dim=16)
    optimizer = optim.Adam(vae_model.parameters(), lr=LR)
    
    # ==========================
    # 4. Train
    # ==========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_vae(data_loader, vae_model, optimizer, device=device, epochs=EPOCHS, log_interval=10)
    
    print("Training complete! You can now use vae_model.encode(...) and vae_model.decode(...) for latent representations.")
