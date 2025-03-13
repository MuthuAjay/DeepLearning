import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    Args:
        dim: Dimension of the embedding.
        end: Maximum sequence length.
        theta: Base value for the frequency calculation.
    
    Returns:
        Complex tensor with shape [end, dim // 2] for efficient computation.
    """
    # Ensure dim is even
    if dim % 2 != 0:
        raise ValueError(f"Dimension {dim} must be even")
    
    # Create frequencies for each dimension
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Create position indices
    t = torch.arange(end, device=freqs.device)
    
    # Outer product of position indices and frequencies
    freqs = torch.outer(t, freqs)
    
    # Compute complex exponentials: cos(x) + i*sin(x)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    Args:
        xq: Query states tensor of shape [batch_size, seq_len, n_heads, head_dim]
        xk: Key states tensor of shape [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: Complex tensor of shape [seq_len, head_dim/2]
        
    Returns:
        Tuple of (xq_out, xk_out) with the same shape as the input tensors.
    """
    # Extract shapes
    batch, seq_len, n_heads, head_dim = xq.shape
    
    # Ensure head_dim is even
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension {head_dim} must be even")
    
    # Reshape inputs to complex-valued tensors
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Extend frequency tensor to match the batch and heads dimensions
    freqs_cis = freqs_cis[:seq_len]
    
    # Apply rotation using complex multiplication
    xq_out = torch.view_as_real(xq_complex * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2)
    
    # Return the rotated tensors with original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding implementation as a PyTorch module.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(self.dim, self.max_seq_len, self.base)
        )
        
    def forward(self, q, k):
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            
        Returns:
            Tuple of (q, k) with rotary embeddings applied
        """
        return apply_rotary_emb(q, k, self.freqs_cis)


# Example usage in a self-attention layer
class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project to query, key, value
        q:torch.Tensor = self.q_proj(hidden_states)
        k:torch.Tensor = self.k_proj(hidden_states)
        v:torch.Tensor = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = self.rotary_emb(q, k)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Get weighted sum
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_size
        )
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output


# Example usage
if __name__ == "__main__":
    # Model parameters
    batch_size = 2
    seq_length = 10
    hidden_size = 512
    num_heads = 8
    
    # Create random input
    hidden_states = torch.rand(batch_size, seq_length, hidden_size)
    
    # Initialize the self-attention layer with RoPE
    self_attn = SelfAttentionWithRoPE(hidden_size, num_heads)
    
    # Forward pass
    output = self_attn(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    
    # We can verify the output shape matches the input shape
    assert output.shape == hidden_states.shape