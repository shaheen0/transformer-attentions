import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model must be divisible by num_heads"
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.Output = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        B, T, C = x.shape
        assert C == self.d_model

        Q = self.Q(x).view(B, T, self.n_head, self.head_dim).transpose(-2, -3)
        K = self.K(x).view(B, T, self.n_head, self.head_dim).transpose(-2, -3)
        V = self.V(x).view(B, T, self.n_head, self.head_dim).transpose(-2, -3)
        # Q, K, V are now (B, n_head, T, head_dim)
        #FlashAttention doesn't change your architecture, it only replaces how the score/softmax/weighted-sum arithmetic gets executed.
        Context = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        # fused kernel: scale, matmul, softmax, matmul all happen inside one
        # GPU kernel, the (T,T) score matrix never gets written to VRAM

        context_matrix = Context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.Output(context_matrix)