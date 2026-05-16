"""
model.py - Transformer implementation for DA6401 Assignment 3.
"""

import copy
import math
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention."""
    d_k = Q.size(-1)
    use_scaling = getattr(scaled_dot_product_attention, "use_scaling", True)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Create encoder padding mask."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Create decoder padding and causal mask."""
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )
    return pad_mask | causal_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    )


class MultiHeadAttention(nn.Module):
    """Manual multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_scaling = getattr(scaled_dot_product_attention, "use_scaling", True)
        scaled_dot_product_attention.use_scaling = self.use_scaling
        try:
            Q = self._split_heads(self.W_q(query))
            K = self._split_heads(self.W_k(key))
            V = self._split_heads(self.W_v(value))
            attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        finally:
            scaled_dot_product_attention.use_scaling = original_scaling

        self.attn_weights = attn_weights
        output = self._combine_heads(attn_output)
        return self.W_o(output)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding used for ablation experiments."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class PositionwiseFeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder block with post-layer normalization."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout, use_scaling=use_scaling
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """Single decoder block with post-layer normalization."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout, use_scaling=use_scaling
        )
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout, use_scaling=use_scaling
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer."""

    def __init__(
        self,
        src_vocab_size: int = 8000,
        tgt_vocab_size: int = 6000,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 1,
        src_itos: Optional[dict] = None,
        tgt_itos: Optional[dict] = None,
        tgt_stoi: Optional[dict] = None,
        src_stoi: Optional[dict] = None,
        src_tokenizer: Optional[Callable] = None,
        checkpoint_path: Optional[str] = None,
        use_learned_positional_encoding: bool = False,
        attention_use_scaling: bool = True,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.pad_idx = pad_idx
        self.use_learned_positional_encoding = use_learned_positional_encoding
        self.attention_use_scaling = attention_use_scaling
        self.max_len = max_len

        self.src_itos = src_itos or {}
        self.tgt_itos = tgt_itos or {}
        self.tgt_stoi = tgt_stoi or {}
        self.src_stoi = src_stoi or {}
        self.src_tokenizer = src_tokenizer

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        pos_cls = (
            LearnedPositionalEncoding
            if use_learned_positional_encoding
            else PositionalEncoding
        )
        self.src_positional_encoding = pos_cls(
            d_model, dropout=dropout, max_len=max_len
        )
        self.tgt_positional_encoding = pos_cls(
            d_model, dropout=dropout, max_len=max_len
        )

        encoder_layer = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_scaling=attention_use_scaling,
        )
        decoder_layer = DecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_scaling=attention_use_scaling,
        )
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_linear.weight = self.tgt_embed.weight

        self._reset_parameters()

        if checkpoint_path is not None:
            import gdown

            gdown.download(
                id="1bb6TTk1Bgl2Rf9IfmpnH8RTxqABIpYF5",
                output=checkpoint_path,
                quiet=False,
            )
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.load_state_dict(state)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_embeddings = self.src_embed(src) * math.sqrt(self.d_model)
        src_embeddings = self.src_positional_encoding(src_embeddings)
        return self.encoder(src_embeddings, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_embeddings = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embeddings = self.tgt_positional_encoding(tgt_embeddings)
        decoded = self.decoder(tgt_embeddings, memory, src_mask, tgt_mask)
        return self.output_linear(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        if (
            self.src_tokenizer is None
            or not self.src_stoi
            or not self.tgt_itos
            or not self.tgt_stoi
        ):
            raise ValueError(
                "Tokenizer and vocab metadata must be attached for inference."
            )

        from train import greedy_decode

        device = next(self.parameters()).device
        if hasattr(self.src_tokenizer, "tokenizer"):
            tokens = [
                tok.text.lower() for tok in self.src_tokenizer.tokenizer(src_sentence)
            ]
        else:
            tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        tokens = ["<sos>"] + tokens[: self.max_len - 2] + ["<eos>"]
        src_ids = [
            self.src_stoi.get(token, self.src_stoi.get("<unk>", 0)) for token in tokens
        ]

        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src_tensor, pad_idx=self.pad_idx)
        decoded = greedy_decode(
            self,
            src_tensor,
            src_mask,
            max_len=self.max_len,
            start_symbol=self.tgt_stoi.get("<sos>", 2),
            end_symbol=self.tgt_stoi.get("<eos>", 3),
            device=device,
        )

        special_tokens = {
            self.tgt_stoi.get("<sos>", 2),
            self.tgt_stoi.get("<eos>", 3),
            self.tgt_stoi.get("<pad>", self.pad_idx),
        }
        translated_tokens = [
            self.tgt_itos[idx]
            for idx in decoded.squeeze(0).tolist()
            if idx not in special_tokens and idx in self.tgt_itos
        ]
        return " ".join(translated_tokens).strip()
