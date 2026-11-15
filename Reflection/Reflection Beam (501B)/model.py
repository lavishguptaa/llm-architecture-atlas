import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer


def compute_rope_params(head_dim, theta_base, context_length):

    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )

    positions = torch.arange(context_length).float()

    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape

    cos = cos[:seq_len, :].to(x.device, dtype=x.dtype)
    sin = sin[:seq_len, :].to(x.device, dtype=x.dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


def position_query_scale(start_pos, num_tokens, device):
    # Scalable-softmax query scale (1 + (position + 1) / 2048) ** 0.1, used by the layers
    # that have no positional encoding: it grows slowly with the position, which keeps the
    # attention sharp at long range.
    positions = torch.arange(start_pos, start_pos + num_tokens, device=device).float()
    scale = torch.exp(0.1 * torch.log1p((positions + 1) / 2048))
    return scale[None, None, :, None]


def depth_scaled_residual_add(x, shortcut, add_idx):
    # The k-th residual addition of the network (k = 2, 3, 4, ... counting attention and
    # feed-forward separately) keeps the variance of the stream constant:
    # x / sqrt(k) + shortcut * sqrt((k - 1) / k)
    return x / math.sqrt(add_idx) + shortcut * math.sqrt((add_idx - 1) / add_idx)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # The normalisation runs in float32, the result goes back to the input dtype
        input_dtype = x.dtype
        x = x.float()

        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )

        x = x / rms

        return (self.weight * x).to(input_dtype)


class GroupedQueryAttention(nn.Module):
    # Two kinds of layer share this code:
    #   sliding_attention: RoPE, every token sees only the last `sliding_window` tokens
    #   full_attention   : no positional encoding at all (NoPE); queries are scaled by a
    #                      position-dependent factor instead
    # Both have per-head QK-norm and a sigmoid output gate for every head.
    def __init__(self, cfg, is_swa):
        super().__init__()

        d_in = cfg["hidden_size"]

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.attention_dropout_p = cfg["attention_dropout"]
        self.scale = self.head_dim ** -0.5

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.sliding_window = cfg["sliding_window"] if is_swa else None

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_in, bias=bias)

        # Normalisation over each head's own dimension (not over the whole projection)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])

        # Output gate: one value per head and channel, plus one bias per head
        self.gate_proj = nn.Linear(d_in, self.num_heads * self.head_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads))

    def project(self, x):
        batch_size, num_tokens, _ = x.shape

        Q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, num_tokens, self.num_key_value_heads, self.head_dim)

        Q = self.q_norm(Q).transpose(1, 2)
        K = self.k_norm(K).transpose(1, 2)
        V = V.transpose(1, 2)

        return Q, K, V

    def attend(self, Q, K, V, past_len=0):
        # Q: [batch, heads, new_tokens, head_dim]; K, V cover past_len + new_tokens positions
        num_tokens = Q.size(2)
        total = K.size(2)

        mask = None

        if self.sliding_window is not None or past_len > 0:
            q_pos = torch.arange(past_len, past_len + num_tokens, device=Q.device)[:, None]
            k_pos = torch.arange(total, device=Q.device)[None, :]

            mask = k_pos <= q_pos

            if self.sliding_window is not None:
                mask = mask & (k_pos > q_pos - self.sliding_window)

        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            is_causal=mask is None,
            scale=self.scale,
            enable_gqa=True
        )

    def finish(self, x, context):
        batch_size, num_tokens, _ = x.shape

        # context: [batch, heads, tokens, head_dim]. Every head's output is multiplied by
        # sigmoid(gate), where the gate is computed from the layer input.
        gate = self.gate_proj(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        gate = gate.transpose(1, 2) + self.gate_bias[None, :, None, None]

        context = context * torch.sigmoid(gate)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.num_heads * self.head_dim
        )

        return self.out_proj(context)

    def forward(self, x, cos, sin, query_scale):
        Q, K, V = self.project(x)

        if self.sliding_window is not None:
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)
        else:
            Q = (Q.float() * query_scale).to(Q.dtype)

        context = self.attend(Q, K, V)

        return self.finish(x, context)


class FeedForward(nn.Module):
    # SwiGLU. Used for the dense layers and as every routed expert.
    # The activation and the product are computed in float32.
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = (F.silu(gate.float()) * up.float()).to(gate.dtype)

        return self.down_proj(hidden)


class MoEFeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["hidden_size"]
        self.num_experts = cfg["num_experts"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        if self.num_experts % self.n_group != 0:
            raise ValueError("num_experts must be divisible by n_group")

        self.gate = nn.Linear(self.emb_dim, self.num_experts, bias=False)

        # Bias used only to pick the experts, never to weight them
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.num_experts),
        )

        self.experts = nn.ModuleList(
            [
                FeedForward(self.emb_dim, cfg["moe_intermediate_size"])
                for _ in range(self.num_experts)
            ]
        )

    def route(self, x_flat):
        # The router always runs in float32; sigmoid scores, not softmax
        scores = torch.sigmoid(F.linear(x_flat.float(), self.gate.weight.float()))

        choice_scores = scores + self.e_score_correction_bias

        # Group-limited routing: keep only the best topk_group groups of experts
        num_tokens = scores.shape[0]
        group_scores = (
            choice_scores
            .view(num_tokens, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, self.topk_group, dim=-1)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = (
            group_mask
            .unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, self.num_experts)
        )
        choice_scores = choice_scores.masked_fill(score_mask == 0, float("-inf"))

        topk_indices = torch.topk(choice_scores, self.num_experts_per_tok, dim=-1)[1]

        # Mixing weights come from the original scores, without the bias
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        return topk_weights, topk_indices

    def forward(self, x):
        batch, seq_len, _ = x.shape

        x_flat = x.reshape(batch * seq_len, self.emb_dim)

        topk_weights, topk_indices = self.route(x_flat)
        topk_weights = topk_weights.to(x_flat.dtype)

        routed_out = torch.zeros_like(x_flat)

        # Run only the experts that received tokens
        for expert_id in torch.unique(topk_indices).tolist():
            token_idx, slot_idx = torch.where(topk_indices == expert_id)

            expert_out = self.experts[expert_id](x_flat[token_idx])
            expert_out = expert_out * topk_weights[token_idx, slot_idx].unsqueeze(-1)

            routed_out.index_add_(0, token_idx, expert_out)

        # The scaling factor is applied once, after the experts are summed. There is no shared expert.
        return (routed_out * self.routed_scaling_factor).reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.is_swa = cfg["layer_types"][layer_idx] == "sliding_attention"

        self.attn = GroupedQueryAttention(cfg, self.is_swa)

        if cfg["mlp_layer_types"][layer_idx] == "sparse":
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = FeedForward(cfg["hidden_size"], cfg["intermediate_size"])

        # Four norms per block: one before and one after each of the two sub-layers
        self.input_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attn_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.pre_ffn_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_ffn_norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

        # Which residual addition of the whole network these two are (see depth_scaled_residual_add)
        self.attn_add_idx = 2 * (layer_idx + 1)
        self.ffn_add_idx = self.attn_add_idx + 1

    def forward(self, x, cos, sin, query_scale):
        shortcut = x
        x = self.input_norm(x)
        x = self.attn(x, cos, sin, query_scale)
        x = self.post_attn_norm(x)
        x = depth_scaled_residual_add(x, shortcut, self.attn_add_idx)

        shortcut = x
        x = self.pre_ffn_norm(x)
        x = self.ffn(x)
        x = self.post_ffn_norm(x)
        x = depth_scaled_residual_add(x, shortcut, self.ffn_add_idx)

        return x


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        rope = cfg["rope_parameters"]
        if rope["full_attention"] is not None:
            raise ValueError("full-attention layers are expected to have no RoPE")
        if rope["sliding_attention"]["rope_type"] != "default":
            raise ValueError("only the default (unscaled) RoPE is supported")

        if len(cfg["layer_types"]) != cfg["num_hidden_layers"] or len(cfg["mlp_layer_types"]) != cfg["num_hidden_layers"]:
            raise ValueError("layer_types and mlp_layer_types need one entry per layer")

        self.final_logit_softcapping = cfg["final_logit_softcapping"]

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        if cfg["tie_word_embeddings"]:
            self.out_proj.weight = self.tok_emb.weight

        # RoPE is only used by the sliding-window layers
        cos, sin = compute_rope_params(
            cfg["head_dim"], rope["sliding_attention"]["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        x = self.tok_emb(idx)

        query_scale = position_query_scale(0, idx.size(1), idx.device)

        for block in self.blocks:
            x = block(x, self.cos, self.sin, query_scale)

        x = self.out_norm_layer(x)
        logits = self.out_proj(x)

        # Soft-cap the logits to (-cap, cap), computed in float32
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits.float() / cap) * cap

        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main():

    with open("config.json") as f:
        cfg = json.load(f)

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()

    input_text = "Hello, how are you?"

    tokenizer = Tokenizer.from_file("tokenizer.json")

    input_ids = tokenizer.encode(input_text).ids
    encoded_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=cfg["max_position_embeddings"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
