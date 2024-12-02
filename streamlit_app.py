import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

# Custom dataset class for vocabulary and utilities
class NameDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.names = [line.strip() + " " for line in f.readlines()]
        self.chars = sorted(list(set("".join(self.names))))
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

# Define the model
class NameClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(NameClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        outputs, hidden = self.lstm(embedded, hidden)
        out = self.fc(outputs)
        return out, hidden

# Load the datasets for vocab information
male_dataset = NameDataset("vyru_vardai.txt")
female_dataset = NameDataset("moteru_vardai.txt")

# Load models
male_model = NameClassifier(male_dataset.vocab_size, embed_dim=16, hidden_dim=32)
female_model = NameClassifier(female_dataset.vocab_size, embed_dim=16, hidden_dim=32)

male_model.load_state_dict(torch.load("male_name_classifier.pth"))
female_model.load_state_dict(torch.load("female_name_classifier.pth"))

male_model.eval()
female_model.eval()

# Name generation function
def generate_name(model, dataset, start_str, max_len=20):
    if not start_str:  # If input is empty, start with a random character
        start_str = random.choice(list(dataset.char_to_int.keys()))
    start_str = start_str.lower()
    input_seq = torch.tensor([dataset.char_to_int[char] for char in start_str if char in dataset.char_to_int]).unsqueeze(0)
    hidden = None
    generated_name = start_str

    for _ in range(max_len - len(start_str)):
        with torch.no_grad():
            outputs, hidden = model(input_seq, hidden)
            outputs = outputs[:, -1, :]  # Take the last time step
            probs = torch.softmax(outputs, dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = dataset.int_to_char[next_char_idx]

            if next_char == " ":  # End of name
                break
            generated_name += next_char
            input_seq = torch.tensor([[next_char_idx]])

    return generated_name.capitalize()

# Streamlit UI
st.title("Name Generator")
st.write("Generate names based on gender and starting characters.")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Input: starting characters
start_str = st.text_input("Enter the starting character(s) (leave empty for random):", max_chars=5)

# Input: gender selection
gender = st.selectbox("Select gender:", ("Male", "Female"))

# Generate button
if st.button("Generate Name"):
    model = male_model if gender == "Male" else female_model
    dataset = male_dataset if gender == "Male" else female_dataset
    generated_name = generate_name(model, dataset, start_str)
    st.session_state.history.append((gender, generated_name))
    st.success(f"Generated Name: {generated_name}")

# Display history
if st.session_state.history:
    st.subheader("History of Generated Names")
    for gender, name in reversed(st.session_state.history):
        st.write(f"{gender}: {name}")
