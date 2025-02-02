import os
import csv
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from TextEncoder import SmallTransformerLM
from TextEncoder import tokenize_text, build_vocab

# ========== 1. Load VAE (Frozen) ==========

class VAESimple(nn.Module):
    """
    This matches the architecture from Step 2: same input size, same latent_dim.
    We'll re-instantiate it here and load weights from 'vae.pth'.
    """
    def __init__(self, latent_dim=16):
        super(VAESimple, self).__init__()
        self.latent_dim = latent_dim
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.flatten_size = 128 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim * 8 * 8)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim * 8 * 8)
        
        # DECODER
        self.decoder_input = nn.Linear(latent_dim * 8 * 8, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 128, 8, 8)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    # Helper to get deterministic latent from x (bypasses sampling)
    def encode_to_latent(self, x):
        with torch.no_grad():
            mu, log_var = self.encode(x)
            # To keep it deterministic, let's just use mu (or sample if you prefer)
            z = mu
        return z
    
    def decode_from_latent(self, z):
        with torch.no_grad():
            return self.decode(z)

# ========== 3. Diffusion Dataset ==========

class ShapeTextLatentDataset(Dataset):
    """
    This dataset will:
    1) Load the 64x64 image from CSV
    2) Convert to VAE latent (freeze VAE).
    3) Convert text to token IDs & embed with the text encoder OR just store token IDs 
       (here we'll store the token IDs, do the embedding in the training loop).
    
    We'll do the embedding in the training step to illustrate dynamic cross-attention/embedding usage.
    """
    def __init__(
        self, 
        csv_file,
        root_dir,
        transform,
        vae,
        text_tokenize_fn,
        word2idx,
        max_text_len=10
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.vae = vae
        self.text_tokenize_fn = text_tokenize_fn
        self.word2idx = word2idx
        self.max_text_len = max_text_len
        
        self.samples = []
        with open(csv_file, "r", encoding="utf-8") as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) < 2:
                    continue
                img_path = parts[0].strip()
                text_desc = parts[1].strip()
                self.samples.append((img_path, text_desc))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_filename, text_desc = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Freeze VAE usage: convert image to latent
        with torch.no_grad():
            # shape: (3, 64, 64) -> (batch=1, 3, 64, 64)
            image = image.unsqueeze(0).to(device = self.vae.fc_mu.weight.device)
            mu, log_var = self.vae.encode(image)  # shape: (1, latent_dim*8*8)
            # Deterministic approach: use mu
            z = mu  
            # Reshape to (latent_dim, 8, 8)
            # because in this example, vae.fc_mu outputs (batch, latent_dim * 8 * 8)
            # We'll keep it as a 2D feature map
            latent_dim = self.vae.latent_dim
            z = z.view(latent_dim, 8, 8)
        
        # convert text to token IDs (we do embedding in train loop)
        token_ids = self.text_tokenize_fn(text_desc, self.word2idx, self.max_text_len)
        
        # shape: (latent_dim, 8, 8), token_ids: list[int]
        return z.squeeze(0), torch.tensor(token_ids, dtype=torch.long)

# ========== 4. Simple Diffusion U-Net ==========

class SimpleUNetRefined(nn.Module):
    """
    A small U-Net that operates on (latent_dim, 8, 8) feature maps,
    plus conditioning from time + text embeddings by channel-wise concatenation.
    """
    def __init__(
        self,
        latent_dim=16,
        text_embed_dim=128
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_embed_dim = text_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Text embedding transform
        self.text_mapper = nn.Sequential(
            nn.Linear(self.text_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Pre-conv to merge (latent_dim + 32) -> 64
        # Actually we have (latent_dim) + 32 from time + 32 from text = 64
        # So total extra is 64, plus z_t's 16 => 16+64=80. We'll define preconv to handle 80->64
        in_channels = latent_dim + 64
        self.pre_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # U-Net down
        self.down1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 8->4
        self.down2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1) # 4->2
        
        # mid
        self.mid = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # U-Net up
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1) # 2->4
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4->8
        
        # final conv
        self.final_conv = nn.Conv2d(64, latent_dim, kernel_size=3, padding=1)
        
        self.act = nn.ReLU()
    
    def forward(self, z_t, t, text_emb):
        """
        z_t: (batch, latent_dim, 8, 8)
        t: (batch,) int for the timestep (0..T-1)
        text_emb: (batch, text_embed_dim)
        
        returns: (batch, latent_dim, 8, 8) predicted noise or predicted z_0
        """
        b, c, h, w = z_t.shape
        
        # 1) time embedding (normalize t for small values)
        t = t.float().unsqueeze(-1) / 100.0
        t_emb = self.time_embed(t)  # (batch, 32)
        
        # 2) text embedding
        txt_e = self.text_mapper(text_emb)  # (batch, 32)
        
        # 3) combine
        cond = t_emb + txt_e  # (batch, 32)
        
        # We'll expand cond to (batch, 64, h, w)
        # But we have 32 from t_emb + 32 from txt_e => 32 is after addition so that doesn't quite make sense
        # Actually let's just do cat: cat(t_emb, txt_e, dim=-1) => shape (batch, 64)
        # We'll do that:
        cond_cat = torch.cat([t_emb, txt_e], dim=-1)  # (batch, 64)
        cond_cat = cond_cat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        
        # cat with z_t => shape (batch, latent_dim+64, 8, 8)
        x = torch.cat([z_t, cond_cat], dim=1)
        
        x = self.pre_conv(x)  # => (batch, 64, 8, 8)
        
        # U-Net down
        d1 = self.act(self.down1(x))  # (batch, 128, 4, 4)
        d2 = self.act(self.down2(d1)) # (batch, 128, 2, 2)
        
        # mid
        m = self.act(self.mid(d2))    # (batch, 128, 2, 2)
        
        # U-Net up
        u2 = self.act(self.up2(m))    # => (batch, 128, 4, 4)
        # skip connection (optionally cat with d1 if you want, but let's keep it simple)
        u1 = self.act(self.up1(u2))   # => (batch, 64, 8, 8)
        
        out = self.final_conv(u1)     # => (batch, latent_dim, 8, 8)
        
        return out

# ========== 5. Diffusion Utilities ==========

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Creates a linearly increasing schedule of beta values (noise scales)
    for the diffusion process.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

class LatentDiffusionTrainer:
    """
    Trains a simple DDPM-like diffusion on VAE latents, 
    with text conditioning.
    """
    def __init__(self, unet, text_encoder, timesteps=100, device = "cuda"):
        self.unet = unet
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.device = device
        
        # Precompute betas, alphas, etc.
        self.betas = linear_beta_schedule(timesteps, 1e-4, 0.02).to(device)
        self.alphas = 1.0 - self.betas.to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
    
    def get_index_from_list(self, vals, t, x_shape):
        """
        For each sample in batch, get vals[t[i]] and reshape to x_shape.
        """
        batch_size = t.size(0)
        out = vals[t].to(t.device)
        return out.view(batch_size, *([1]*(len(x_shape)-1)))
    
    def forward_diffusion_sample(self, z0, t, noise=None):
        """
        q(z_t | z_0) = sqrt(alpha_bar_t)*z0 + sqrt(1-alpha_bar_t)*noise
        """
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha_bar_t = self.get_index_from_list(self.alpha_bars, t, z0.shape)**0.5
        sqrt_one_minus_alpha_bar_t = (1 - self.get_index_from_list(self.alpha_bars, t, z0.shape))**0.5
        z_t = sqrt_alpha_bar_t * z0 + sqrt_one_minus_alpha_bar_t * noise
        return z_t, noise
    
    def train_step(self, z0, text_tokens, optimizer, device):
        """
        Single training step:
        1) pick t
        2) forward diffuse z0 -> zt
        3) unet predicts noise
        4) compute loss
        """
        # sample a random t for each sample
        bsz = z0.size(0)
        t = torch.randint(0, self.timesteps, (bsz,), device=device).long()
        
        # embed text
        with torch.no_grad():
            text_emb = self.text_encoder.encode_text(text_tokens.to(device)) 
            # (batch, text_embed_dim)
        
        z0 = z0.to(device)
        
        # forward diffusion
        z_t, noise = self.forward_diffusion_sample(z0, t)
        
        # predict noise
        noise_pred = self.unet(z_t, t, text_emb)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

# ========== 6. Putting It All Together ==========

if __name__ == "__main__":
    ########## CONFIG ##########
    ROOT_DIR = "C:\Data"
    CSV_FILE = os.path.join(ROOT_DIR,"trainingData.csv")
    VAE_PATH = "vae.pth"
    TEXT_ENCODER_PATH = "text_encoder.pth"
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 1e-4
    TIMESTEPS = 100
    ###########################
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1) Load the frozen VAE
    vae = VAESimple(latent_dim=16)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device)
    print("Loaded and froze VAE.")
    
    # 2) Load/freeze text encoder
    word2idx, idx2word = build_vocab(CSV_FILE)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    text_encoder = SmallTransformerLM(vocab_size=vocab_size, d_model=128, n_heads=4, num_layers=2, max_len=10)
    text_encoder.load_state_dict(torch.load(TEXT_ENCODER_PATH, map_location=device))
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.to(device)
    print("Loaded and froze Text encoder.")
    
    # 3) Create dataset & dataloader
    # We'll use a transform that does basic to-tensor
    dataset_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])
    
    # We'll pass the VAE into the dataset so it can produce latents
    dataset = ShapeTextLatentDataset(
        csv_file=CSV_FILE,
        root_dir=ROOT_DIR,
        transform=dataset_transform,
        vae=vae,
        text_tokenize_fn=tokenize_text,
        word2idx=word2idx,
        max_text_len=10
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("Dataset ready.")
    
    # 4) Initialize the U-Net in latent space
    unet = SimpleUNetRefined(latent_dim=16, text_embed_dim=128).to(device)
    
    # 5) Diffusion trainer
    diffusion_trainer = LatentDiffusionTrainer(
        unet=unet,
        text_encoder=text_encoder,
        timesteps=TIMESTEPS
    )
    
    # 6) Optimizer
    optimizer = optim.Adam(unet.parameters(), lr=LR)
    
    # 7) Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (z0, text_tokens) in enumerate(dataloader):
            # z0 shape: (batch, latent_dim, 8, 8)
            # text_tokens shape: (batch, seq_len=10)
            loss_val = diffusion_trainer.train_step(z0, text_tokens, optimizer, device)
            total_loss += loss_val
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
    
    # 8) Save the diffusion model
    torch.save(unet.state_dict(), "latent_diffusion_unet.pth")
    print("Diffusion model training complete and saved.")
