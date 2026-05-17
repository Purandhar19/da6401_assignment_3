"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import copy
import math
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, broadcastable to (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT (set to -inf).

    Returns:
        output : shape (..., seq_q, d_v)
        attn_w : shape (..., seq_q, seq_k)  — sums to 1 along last dim
    """
    d_k = Q.size(-1)
    use_scaling = getattr(scaled_dot_product_attention, "use_scaling", True)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (softmax of all -inf) with 0
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder.

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → PAD token (masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal mask for the decoder.

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → masked out (PAD or future position)
    """
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    )
    return pad_mask | causal_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    )


def _detokenize_text(text: str) -> str:
    replacements = {
        " .": ".",
        " ,": ",",
        " !": "!",
        " ?": "?",
        " :": ":",
        " ;": ";",
        " n't": "n't",
        " 's": "'s",
        " 're": "'re",
        " 've": "'ve",
        " 'm": "'m",
        " 'll": "'ll",
        " 'd": "'d",
        "( ": "(",
        " )": ")",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — torch.nn.MultiheadAttention is NOT used.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        original_scaling = getattr(scaled_dot_product_attention, "use_scaling", True)
        scaled_dot_product_attention.use_scaling = self.use_scaling
        try:
            attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        finally:
            scaled_dot_product_attention.use_scaling = original_scaling
        self.attn_weights = attn_weights  # stored for visualisation

        return self.W_o(self._combine_heads(attn_output))


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding — pe registered as a non-trainable buffer.
    """

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
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for ablation experiments."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW₁+b₁)W₂+b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════


class EncoderLayer(nn.Module):
    """Self-Attn → Add&Norm → FFN → Add&Norm  (Post-LN)"""

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════


class DecoderLayer(nn.Module):
    """Masked-Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm"""

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════


class Encoder(nn.Module):
    """Stack of N identical EncoderLayers + final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayers + final LayerNorm."""

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


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    src_vocab_size and tgt_vocab_size have defaults so the autograder
    can call Transformer() with no arguments.
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
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
        checkpoint_path: str = None,
        max_len: int = 100,
        use_learned_positional_encoding: bool = False,
        attention_use_scaling: bool = True,
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
        self.max_len = max_len
        self.src_itos = src_itos or {}
        self.tgt_itos = tgt_itos or {}
        self.tgt_stoi = tgt_stoi or {}
        self.src_stoi = src_stoi or {}
        self.src_tokenizer = src_tokenizer
        self.use_learned_positional_encoding = use_learned_positional_encoding
        self.attention_use_scaling = attention_use_scaling

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        pos_cls = (
            LearnedPositionalEncoding
            if use_learned_positional_encoding
            else PositionalEncoding
        )
        self.src_positional_encoding = pos_cls(d_model, dropout, max_len=max_len)
        self.tgt_positional_encoding = pos_cls(d_model, dropout, max_len=max_len)

        enc_layer = EncoderLayer(
            d_model, num_heads, d_ff, dropout, use_scaling=attention_use_scaling
        )
        dec_layer = DecoderLayer(
            d_model, num_heads, d_ff, dropout, use_scaling=attention_use_scaling
        )
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_linear.weight = self.tgt_embed.weight  # weight tying

        self._reset_parameters()

        should_autoload_default_checkpoint = (
            checkpoint_path is None
            and not self.src_stoi
            and not self.tgt_stoi
            and src_vocab_size == 7853
            and tgt_vocab_size == 5893
            and d_model == 256
            and N == 3
            and num_heads == 8
            and d_ff == 512
        )
        if should_autoload_default_checkpoint:
            checkpoint_path = "best_model.pt"

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                import gdown

                gdown.download(
                    id="1OWO62R9qm4RImEKhW6RMBFCjojYLNvQn",
                    output=checkpoint_path,
                    quiet=False,
                )
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in state:
                    vocab_metadata = state.get("vocab_metadata", {})
                    self.src_stoi = vocab_metadata.get("src_stoi", self.src_stoi)
                    self.tgt_stoi = vocab_metadata.get("tgt_stoi", self.tgt_stoi)
                    self.src_itos = vocab_metadata.get("src_itos", self.src_itos)
                    self.tgt_itos = vocab_metadata.get("tgt_itos", self.tgt_itos)
                    state = state["model_state_dict"]
                self.load_state_dict(state, strict=False)

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ─────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_positional_encoding(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_positional_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.output_linear(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # ── INFERENCE ────────────────────────────────────────────────────

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using greedy decoding.
        Must remain fully offline during autograder evaluation.
        """
        device = next(self.parameters()).device
        src_stoi = (
            self.src_stoi
            if self.src_stoi
            else {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        )
        tgt_stoi = (
            self.tgt_stoi
            if self.tgt_stoi
            else {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
        )
        tgt_itos = (
            self.tgt_itos
            if self.tgt_itos
            else {idx: tok for tok, idx in tgt_stoi.items()}
        )

        if self.src_tokenizer is not None:
            if hasattr(self.src_tokenizer, "tokenizer"):
                src_tokens = [
                    tok.text.lower()
                    for tok in self.src_tokenizer.tokenizer(src_sentence)
                ]
            else:
                src_tokens = [
                    tok.text.lower() for tok in self.src_tokenizer(src_sentence)
                ]
        else:
            src_tokens = src_sentence.lower().strip().split()

        tokens = ["<sos>"] + src_tokens[: self.max_len - 2] + ["<eos>"]
        unk = src_stoi.get("<unk>", 0)
        src_ids = [src_stoi.get(t, unk) for t in tokens]
        src_t = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src_t, pad_idx=self.pad_idx)

        sos_idx = tgt_stoi.get("<sos>", 2)
        eos_idx = tgt_stoi.get("<eos>", 3)

        self.eval()
        with torch.no_grad():
            memory = self.encode(src_t, src_mask)
            ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            for _ in range(self.max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                out = self.decode(memory, src_mask, ys, tgt_mask)
                nxt = int(out[:, -1, :].argmax(dim=-1).item())
                ys = torch.cat(
                    [ys, torch.tensor([[nxt]], dtype=torch.long, device=device)],
                    dim=1,
                )
                if nxt == eos_idx:
                    break

        skip = {sos_idx, eos_idx, tgt_stoi.get("<pad>", 1)}
        return _detokenize_text(
            " ".join(
                tgt_itos[i]
                for i in ys.squeeze(0).tolist()
                if i not in skip and i in tgt_itos
            )
        )
