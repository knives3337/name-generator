import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Create a custom dataset class
class NameDataset(Dataset):
    def __init__(self, file_path):
        # Load names from file and add padding character
        with open(file_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() + " " for line in f.readlines()]
        # Create character dictionaries
        self.chars = sorted(list(set("".join(self.names))))  # Unique characters
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name), len(encoded_name)

# Collate function for padding
def collate_fn(batch):
    names, lengths = zip(*batch)
    max_len = max(lengths)
    padded_names = torch.zeros((len(names), max_len), dtype=torch.long)
    for i, name in enumerate(names):
        padded_names[i, :lengths[i]] = name
    return padded_names, torch.tensor(lengths)

# Define the model
class NameClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NameClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)  # Get outputs for all time steps
        out = self.fc(outputs)  # Pass through fully connected layer
        return out


# Training function
def train_model(dataset, output_file, embed_dim=16, hidden_dim=32, batch_size=32, epochs=200, lr=0.001):
    # Prepare data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model, loss, and optimizer
    model = NameClassifier(dataset.vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch, lengths in dataloader:
            if batch.size(0) == 0:  # Skip empty batches
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)  # Shape: [batch_size, seq_len, vocab_size]
            outputs = outputs[:, :-1, :].contiguous().view(-1, dataset.vocab_size)  # Exclude last time step and flatten
            
            # Prepare targets
            targets = batch[:, 1:].contiguous().view(-1)  # Shift targets and flatten
            
            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate epoch loss
            epoch_loss += loss.item()
        
        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), output_file)
    print(f"Model saved as {output_file}")


# Load datasets
male_dataset = NameDataset("vyru_vardai.txt")
female_dataset = NameDataset("moteru_vardai.txt")

# Train separate models for male and female names
train_model(male_dataset, "male_name_classifier.pth")
train_model(female_dataset, "female_name_classifier.pth")
