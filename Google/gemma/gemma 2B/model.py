import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length, stride):
        
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(data).ids
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
                
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(data, tokenizer, context_size=256, stride=128,
                      batch_size=8, shuffle=True, num_workers=0, drop_last=False):
    
    if tokenizer is None:
        raise ValueError("tokenizer must be provided; load tokenizer.json before creating the dataloader")
    
    dataset = CustomDataset(data, tokenizer, context_size, stride)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    
    return dataloader

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

    # Half-split pairing (dimension i with i + head_dim / 2)
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotated = torch.cat([-x2, x1], dim=-1)

    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_heads = cfg["num_attention_heads"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        d_in = cfg["hidden_size"]
        d_out = cfg["hidden_size"]

        self.attention_dropout = cfg["attention_dropout"]

        bias = cfg["attention_bias"]
        self.W_q = nn.Linear(d_in, self.num_heads * self.head_dim, bias=bias)
        self.W_k = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_in, self.num_key_value_heads * self.head_dim, bias=bias)

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, d_out, bias=bias)

    def forward(self,x,cos,sin):
        batch_size,num_tokens,d_in = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        K = K.view(batch_size,num_tokens,self.num_key_value_heads,self.head_dim)
        V = V.view(batch_size,num_tokens,self.num_key_value_heads,self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        
        
        context = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=(
                   self.attention_dropout
                    if self.training
                    else 0.0
                ),
            is_causal=True,
            enable_gqa=True,
        )
        
        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            num_tokens,
            self.num_heads * self.head_dim,
        )


        return self.out_proj(context)
    
class FeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim, activation):
        super().__init__()

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        # Gemma uses the tanh-approximate GELU ("gelu_pytorch_tanh"); "gelu" selects the exact GELU
        approximations = {"gelu_pytorch_tanh": "tanh", "gelu": "none"}
        if activation not in approximations:
            raise ValueError(f"Unsupported hidden_activation: {activation}")
        self.gelu_approximation = approximations[activation]

    def forward(self, x):
        """Apply the gated feed-forward transformation."""
        gate = F.gelu(self.gate_proj(x), approximate=self.gelu_approximation)
        value = self.up_proj(x)

        x = gate * value

        return self.down_proj(x)
    
class RMSNorm(nn.Module):
    # Gemma scales by (1 + weight), so the weight is zero-initialised
    def __init__(self, emb_dim, eps):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        x_float = x.float()

        rms = torch.sqrt(
            torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps
        )

        x_float = x_float / rms

        return (x_float * (1.0 + self.weight.float())).to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        emb_dim = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]

        self.attn = GroupedQueryAttention(cfg)

        self.layer1 = RMSNorm(emb_dim, eps)
        self.layer2 = RMSNorm(emb_dim, eps)

        self.ffn = FeedForward(emb_dim, cfg["intermediate_size"], cfg["hidden_activation"])

    def forward(self,x,cos,sin):
        shortcut = x
        x = self.layer1(x)
        x = self.attn(x,cos,sin)
        x = x + shortcut
        
        shortcut = x
        x = self.layer2(x)
        x = self.ffn(x)
        x = x + shortcut
        
        return x
    
class Model(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["hidden_size"])

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_hidden_layers"])]
        )

        self.out_norm_layer = RMSNorm(cfg["hidden_size"],eps=cfg["rms_norm_eps"])
        self.out_proj = nn.Linear(cfg["hidden_size"],cfg["vocab_size"],bias=False)

        # Gemma ties the output head to the token embedding
        self.out_proj.weight = self.tok_emb.weight

        # Gemma scales the token embeddings by sqrt(hidden_size)
        self.emb_scale = cfg["hidden_size"] ** 0.5

        # One RoPE table shared by every layer
        head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim, cfg["rope_theta"], cfg["max_position_embeddings"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self,idx):
        batch_size,num_tokens = idx.shape
        
        if num_tokens > self.cos.size(0):
            raise ValueError(
                f"Sequence length {num_tokens} exceeds "
                f"max_position_embeddings {self.cos.size(0)}"
            )

        x = self.tok_emb(idx) * self.emb_scale
        
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.out_norm_layer(x)
        x = self.out_proj(x)
        return x
    
    
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

    # config.json lists hidden_act "gelu", but Gemma uses the tanh-approximate GELU
    cfg.setdefault("hidden_activation", "gelu_pytorch_tanh")

    torch.manual_seed(42)
    model = Model(cfg)
    model.eval()
    
    
    input_text = "Hello, how are you?"
    
    tokenizer_path = "tokenizer.json"

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # tokenizer_vocab_size = tokenizer.get_vocab_size()
    # if tokenizer_vocab_size != cfg["vocab_size"]:
    #     raise ValueError(
    #         f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match "
    #         f"model vocab_size ({cfg['vocab_size']})."
    #     )

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