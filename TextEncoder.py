import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# ===================================
# 1. Tokenization & Vocabulary Setup
# ===================================

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

def build_vocab(csv_file):
    """
    Reads text descriptions from CSV and builds a minimal vocabulary.
    Returns a dict: word2idx and idx2word
    """
    words = set()
    
    # Gather all words from text_description
    with open(csv_file, 'r', encoding='utf-8') as f:
        # skip header
        next(f)
        for line in f:
            # line format: "image_filename,text_description"
            parts = line.strip().split(",", 1)
            if len(parts) < 2:
                continue
            text_desc = parts[1].strip()
            # Basic whitespace split
            for w in text_desc.split():
                words.add(w.lower())
    
    # Convert set to list
    vocab_list = SPECIAL_TOKENS + sorted(list(words))
    
    # Make dictionaries
    word2idx = {w: i for i, w in enumerate(vocab_list)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    return word2idx, idx2word

def tokenize_text(text, word2idx, max_len=10):
    """
    Convert a text (string) into a list of token indices.
    Add [BOS] at start, [EOS] at end, and pad if needed.
    """
    # basic whitespace split
    tokens = text.lower().split()
    
    # Convert tokens to indices, fallback to [UNK]
    token_ids = []
    for w in tokens:
        if w in word2idx:
            token_ids.append(word2idx[w])
        else:
            token_ids.append(word2idx["[UNK]"])
    
    # Add special tokens
    token_ids = [word2idx["[BOS]"]] + token_ids + [word2idx["[EOS]"]]
    
    # Pad or truncate to max_len
    if len(token_ids) < max_len:
        token_ids += [word2idx["[PAD]"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    
    return token_ids

# ===================================
# 2. Dataset & DataLoader
# ===================================
class TextDataset(Dataset):
    """
    Simple dataset that reads the text descriptions from CSV
    and returns tokenized sequences for language modeling.
    """
    def __init__(self, csv_file, word2idx, max_len=10):
        super().__init__()
        self.data = []
        self.word2idx = word2idx
        self.max_len = max_len
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            # skip header
            next(f)
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) < 2:
                    continue
                text_desc = parts[1].strip()
                self.data.append(text_desc)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        token_ids = tokenize_text(text, self.word2idx, self.max_len)
        return torch.tensor(token_ids, dtype=torch.long)

# ===================================
# 3. Transformer Language Model
# ===================================
class SmallTransformerLM(nn.Module):
    """
    A minimal Transformer for next-token prediction on a small vocabulary.
    We'll produce an embedding for each token, then pass it through
    a few layers of self-attention and feed-forward. Finally, a linear head
    predicts the next token. We'll also provide a method to get a final
    sequence embedding (e.g., for text-to-image conditioning).
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=2, max_len=10):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Embedding for tokens + positional encoding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # You can do a learned positional embedding
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer encoder layers (stacked)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=n_heads, 
                                                   dim_feedforward=d_model*4,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=True)  # PyTorch >= 1.10
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len)
        Returns:
          logits for each token in the sequence to predict next token.
          shape: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token + positional embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        token_embeddings = self.token_emb(x)  # (batch, seq_len, d_model)
        pos_embeddings = self.pos_emb(positions)  # (1, seq_len, d_model)
        embeddings = token_embeddings + pos_embeddings
        
        # Pass through Transformer encoder
        encoded = self.transformer_encoder(embeddings)  # (batch, seq_len, d_model)
        
        # Predict next token for each position
        logits = self.lm_head(encoded)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def encode_text(self, x):
        """
        Encode the text into a single vector embedding by taking the final
        hidden state of the last token or an average of all tokens.
        
        x shape: (batch_size, seq_len)
        Returns: (batch_size, d_model) -> final text embeddings
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.pos_emb(positions)
        embeddings = token_embeddings + pos_embeddings
        
        encoded = self.transformer_encoder(embeddings)  # (batch, seq_len, d_model)
        
        # Option A: take the final token embedding
        # final_emb = encoded[:, -1, :]
        
        # Option B: take the mean across all tokens
        final_emb = torch.mean(encoded, dim=1)  # (batch, d_model)
        
        return final_emb

# ===================================
# 4. Training Loop (Next-Token Prediction)
# ===================================
def generate_subsequences(seq):
    """
    For next-token prediction, we can produce input/target pairs:
    If seq = [BOS, red, circle, EOS, PAD, ...]
    input  = [BOS, red, circle, EOS, PAD]
    target = [red, circle, EOS, PAD, ...]
    We shift by one to predict the "next" token.
    """
    input_seq = seq[:-1]
    target_seq = seq[1:]
    return input_seq, target_seq

def train_language_model(model, dataloader, optimizer, device, epochs=5):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignoring [PAD] if it's idx=0
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)  # shape: (batch_size, seq_len)
            
            # create shifted input/target
            inp, tgt = [], []
            for seq in batch:
                i_seq, t_seq = generate_subsequences(seq)
                inp.append(i_seq)
                tgt.append(t_seq)
            
            inp = torch.stack(inp, dim=0)  # (batch, seq_len-1)
            tgt = torch.stack(tgt, dim=0)  # (batch, seq_len-1)
            
            # forward
            logits = model(inp)  # (batch, seq_len-1, vocab_size)
            
            # reshape for cross-entropy
            # We want shape: (batch * seq_len-1, vocab_size) vs (batch * seq_len-1)
            logits_2d = logits.view(-1, logits.size(-1))
            tgt_1d = tgt.view(-1)
            
            loss = criterion(logits_2d, tgt_1d)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.3f}")

# ===================================
# 5. Putting it all together
# ===================================
if __name__ == "__main__":
    # Path to your CSV file from Step 1 (shape dataset)
    CSV_FILE = "C:\Data/trainingData.csv"
    
    # 1. Build Vocab
    word2idx, idx2word = build_vocab(CSV_FILE)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")
    
    # 2. Create Dataset & DataLoader
    dataset = TextDataset(csv_file=CSV_FILE, word2idx=word2idx, max_len=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    
    # 3. Define a small Transformer
    model = SmallTransformerLM(
        vocab_size=vocab_size,
        d_model=128,     # embedding dimension
        n_heads=4,       # number of attention heads
        num_layers=2,    # Transformer encoder layers
        max_len=10       # sequence length (BOS ... tokens ... EOS)
    )
    
    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_language_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=10  # Increase if you want better convergence
    )
    
    # 6. Usage: encode some text
    sample_text = "a large blue circle"
    tokens = tokenize_text(sample_text, word2idx, max_len=10)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens_tensor)
    print(f"Text embedding shape: {embedding.shape}")  # e.g. (1, 128)
    print("Done!")