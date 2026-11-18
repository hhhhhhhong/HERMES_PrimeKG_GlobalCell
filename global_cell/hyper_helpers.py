"""Unmodified MIT author helpers extracted from the pinned HYPER source.
See vendor/HYPER_PROVENANCE.json and vendor/hyper_author/LICENSE.
"""
import torch

def forward_mapping(i,j):
    if i == j: return (i+1)**2 - 1
    if i > j: return (i+1)**2 - 1 - 2*(i-j)
    if i < j: return (j+1)**2 - 2*(j-i)

def sinusoidal_positional_encoding(max_seq_len, dim):
    # Calculate the angle rates based on dimension indices.
    angle_rates = 1 / torch.pow(10000, torch.arange(0, dim, 2).float() / dim)
    # Calculate the angles for each position for half of the dimensions (sine and cosine)
    angles = (torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0))
    # Cosines and sines of the angles to get the RoPE for each position
    position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
    return position_encodings
