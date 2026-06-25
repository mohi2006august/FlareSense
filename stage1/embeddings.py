import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Converts 1-second cadence time series (21,600 steps for 6 hours) 
    into 30-second patches of dimension 256.
    Includes Sinusoidal Positional Encoding and Gap-Aware Masking Token.
    """
    def __init__(self, sequence_length: int = 21600, patch_size: int = 30, embed_dim: int = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = sequence_length // patch_size
        
        # 1D Convolution for Patch Embedding
        # in_channels=1 (flux), out_channels=embed_dim, kernel=patch_size, stride=patch_size
        self.proj = nn.Conv1d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Gap-Aware Masking Token (Learnable Parameter)
        # Replaces patches where all 30 seconds are missing data
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional Encoding buffer (not a learnable parameter)
        self.register_buffer('pos_embedding', self._generate_positional_encoding(self.num_patches, embed_dim))

    def _generate_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """
        Generates standard sinusoidal positional encoding.
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # Shape: (1, seq_len, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, 1, sequence_length) - The preprocessed flux time-series
        mask: (batch_size, sequence_length) - Boolean mask indicating gaps (True if gap)
        
        Returns: (batch_size, num_patches, embed_dim)
        """
        batch_size = x.size(0)
        
        # 1. Patch Projection
        # Input x shape: (B, 1, L)
        # Output x shape: (B, embed_dim, num_patches)
        x = self.proj(x)
        
        # Transpose to (B, num_patches, embed_dim) for standard sequence processing
        x = x.transpose(1, 2)
        
        # 2. Gap-Aware Masking
        if mask is not None:
            # Mask shape is (B, L). We need to aggregate it to patch level.
            # Reshape to (B, num_patches, patch_size) and check if ALL values in a patch are gaps
            mask_patches = mask.view(batch_size, self.num_patches, self.patch_size)
            # A patch is fully masked if all its values are True
            is_patch_masked = mask_patches.all(dim=-1) # Shape: (B, num_patches)
            
            # Expand mask token to match batch size
            expanded_mask_token = self.mask_token.expand(batch_size, -1, -1)
            
            # Apply mask token where is_patch_masked is True
            # Expand is_patch_masked to match embed_dim for torch.where
            is_patch_masked_expanded = is_patch_masked.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            x = torch.where(is_patch_masked_expanded, expanded_mask_token, x)
            
        # 3. Add Positional Encoding
        x = x + self.pos_embedding
        
        return x
