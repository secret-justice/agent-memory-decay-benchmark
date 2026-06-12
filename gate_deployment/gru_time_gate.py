# -*- coding: utf-8 -*-
"""
GRU+Time Memory Gate - drop-in replacement for LTC gate.
Based on v21 benchmark finding: GRU+Time outperforms all ODE models.
"""
import torch
import torch.nn as nn
import numpy as np


class GRUTimeGate(nn.Module):
    """
    GRU+Time gate for memory retrieval decisions.
    Input: query embedding (64-dim) + time features
    Output: keep probability [0, 1]
    """
    
    def __init__(self, input_dim=64, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Time encoding: convert elapsed time to feature vector
        self.time_enc = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, input_dim)
        )
        
        # GRU cell for sequential processing
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        
        # Output head
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_emb, timestamps, context_embs=None):
        """
        Args:
            query_emb: [batch, input_dim] - current query embedding
            timestamps: [batch, 1] - elapsed time since last interaction
            context_embs: [batch, seq_len, input_dim] - optional history
        Returns:
            keep_prob: [batch, 1] - probability of keeping/remembering
        """
        batch_size = query_emb.shape[0]
        
        # Time encoding
        t_enc = self.time_enc(timestamps)  # [batch, input_dim]
        
        # Combine query with time
        x = query_emb + t_enc  # [batch, input_dim]
        
        # Process through GRU
        h = torch.zeros(batch_size, self.hidden_dim, device=query_emb.device)
        
        if context_embs is not None:
            # Process history sequence
            for t in range(context_embs.shape[1]):
                h = self.gru(context_embs[:, t, :], h)
        
        # Final step with current query
        h = self.gru(x, h)
        
        # Output keep probability
        return self.out(h)
    
    def predict_single(self, query_emb, elapsed_time):
        """Single-sample prediction for production use."""
        self.eval()
        with torch.no_grad():
            q = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
            t = torch.tensor([[elapsed_time]], dtype=torch.float32)
            return self(q, t).item()


def create_gate(gate_type="gru_time", **kwargs):
    """Factory function - compatible with existing gate interface."""
    if gate_type == "gru_time":
        return GRUTimeGate(**kwargs)
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
