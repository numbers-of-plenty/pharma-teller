import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os
import time
from safetensors.torch import load_file
import torch.distributed as dist
from torch.amp import autocast

from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_DIR = "/root/Llama-3.2-1B/"
WEIGHTS_PATH = "/root/Llama-3.2-1B/model.safetensors"
CONFIG_PATH = "/root/Llama-3.2-1B/config.json"

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def is_master():
    return get_rank() == 0


def precompute_freqs_cis(dim: int, end: int, theta: float, scaling_config: dict = None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if scaling_config and scaling_config.get("rope_type") == "llama3":
        factor = scaling_config.get("factor", 32.0)
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        original_max_pos = scaling_config.get("original_max_position_embeddings", 8192)
        wavelen = 2 * math.pi / freqs
        smooth = (original_max_pos / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth = torch.clamp(smooth, 0.0, 1.0)
        scaled_freqs = freqs / factor
        freqs = (1 - smooth) * scaled_freqs + smooth * freqs

    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    cos = freqs_cos[: xq.shape[1]].unsqueeze(0).unsqueeze(2)
    sin = freqs_sin[: xq.shape[1]].unsqueeze(0).unsqueeze(2)
    cos = cos.to(xq.device)
    sin = sin.to(xq.device)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed

class LlamaConfig:
    def __init__(self, config_dict):
        self.dim = config_dict["hidden_size"]
        self.n_layers = config_dict["num_hidden_layers"]
        self.n_heads = config_dict["num_attention_heads"]
        self.n_kv_heads = config_dict["num_key_value_heads"]
        self.vocab_size = config_dict["vocab_size"]
        self.intermediate_size = config_dict["intermediate_size"]
        self.norm_eps = config_dict["rms_norm_eps"]
        self.rope_theta = config_dict["rope_theta"]
        self.max_seq_len = config_dict["max_position_embeddings"]
        self.head_dim = self.dim // self.n_heads
        self.rope_scaling = config_dict.get("rope_scaling", {})

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight

class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        B, Seq, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(B, Seq, self.n_heads, self.head_dim)
        xk = xk.view(B, Seq, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, Seq, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)
        
        if self.n_rep > 1:
            B, Seq_kv, _, _ = xk.shape
            xk = xk.unsqueeze(3).expand(B, Seq_kv, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, Seq_kv, self.n_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(B, Seq_kv, self.n_kv_heads, self.n_rep, self.head_dim).reshape(B, Seq_kv, self.n_heads, self.head_dim)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(B, Seq, -1)
        return self.o_proj(output), None

class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, cos, sin, mask, kv_cache=None):
        attn_out, _ = self.self_attn(self.input_layernorm(x), cos, sin, mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, None

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )
        self.register_buffer('freqs_cos_buffer', self.freqs_cos, persistent=False)
        self.register_buffer('freqs_sin_buffer', self.freqs_sin, persistent=False)

    def forward(self, tokens, kv_caches=None, use_cache=False):
        B, Seq = tokens.shape
        h = self.embed_tokens(tokens)

        cos = self.freqs_cos_buffer[:Seq]
        sin = self.freqs_sin_buffer[:Seq]
        mask = torch.full((Seq, Seq), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h, _ = layer(h, cos, sin, mask, None)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

def main():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    if is_master():
        print(f"Loading Config from {CONFIG_PATH}...")
    
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(config_data)

    if is_master():
        print("Initializing the Model on CPU...")
    model = LlamaModel(config)

    if is_master():
        print(f"Loading Weights from {WEIGHTS_PATH}...")
    
    state_dict = load_file(WEIGHTS_PATH)
    final_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]
        final_state_dict[new_key] = value
    if "lm_head.weight" not in final_state_dict:
        final_state_dict["lm_head.weight"] = final_state_dict["embed_tokens.weight"]
    model.load_state_dict(final_state_dict)

    if is_master():
        print("Casting model to bfloat16...")
    model.to(torch.bfloat16)

    if is_master():
        print("Moving Model to GPU for DDP...")
    model.to(device)

    if is_master():
        print("Wrapping with DDP...")
    
    model = DDP(model, device_ids=[local_rank])

    if is_master():
        print(f"Model wrapped with DDP. Running on {dist.get_world_size()} GPUs.")

    if is_master():
        print("\n" + "=" * 60)
        print("STARTING DDP TRAINING BENCHMARK (100 Steps)")
        print("Goal: Measure Tokens/Sec on 2 GPUs")
        print("=" * 60)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    BATCH_SIZE = 4
    SEQ_LEN = 128
    TOTAL_STEPS = 100
    
    dummy_input = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN)).to(device)
    
    if is_master():
        print("Running Warmup Step...")
    logits = model(dummy_input)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), dummy_input.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    
    total_tokens_processed = 0
    start_time_global = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        step_start = time.time()
        
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(dummy_input)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), dummy_input.view(-1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        step_time = time.time() - step_start
        
        world_size = dist.get_world_size()
        global_batch_size = BATCH_SIZE * world_size 
        tokens_per_step = global_batch_size * SEQ_LEN
        total_tokens_processed += tokens_per_step
        
        if is_master():
            if step % 10 == 0 or step == 1:
                tokens_per_sec = tokens_per_step / step_time
                print(f"Batch {step:03d}/{TOTAL_STEPS} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {step_time:.4f}s | "
                      f"Global Speed: {tokens_per_sec:.0f} tokens/s")

    total_time = time.time() - start_time_global
    avg_speed = total_tokens_processed / total_time
    
    if is_master():
        print("=" * 60)
        print(f"DDP BENCHMARK RESULTS ({dist.get_world_size()} GPUs)")
        print(f"Avg Throughput: {avg_speed:.0f} Tokens/Sec")
        print("=" * 60)

    cleanup()

if __name__ == "__main__":
    main()