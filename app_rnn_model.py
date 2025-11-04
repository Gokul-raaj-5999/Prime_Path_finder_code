import torch
import torch.nn as nn
import streamlit as st
import numpy as np

class recomendation_rnn_model(nn.Module):
    def __init__(self, num_movies, embedding_dim=128, hidden_dim=256, gru_layers=2, fc_hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(num_movies, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, num_movies)

    def forward(self, x):
        x = self.embedding(x)
        out, h = self.rnn(x)
        last_hidden = h[-1]
        x = self.fc1(last_hidden)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits
