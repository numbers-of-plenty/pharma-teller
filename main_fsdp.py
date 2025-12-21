import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os
import functools
import time
from safetensors.torch import load_file
from pathlib import Path

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

MODEL_DIR = "/root/Llama-3.2-1B/"
WEIGHTS_PATH = "/root/Llama-3.2-1B/model.safetensors"
CONFIG_PATH = "/root/Llama-3.2-1B/config.json"
TOKENIZER_MODEL = "/root/Llama-3.2-1B/original/tokenizer.model"

try:
    from tokenizer import Tokenizer
except ImportError:
    print("Warning: 'tokenizer.py' not found. Ensure the Tokenizer class is available.")
    Tokenizer = None


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
    # Ensure compat with FSDP devices
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

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)
        
        new_cache = (xk, xv) 

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
        return self.o_proj(output), new_cache

class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x, cos, sin, mask, kv_cache=None):
        attn_out, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )
        # Register buffers so FSDP handles device placement
        self.register_buffer('freqs_cos_buffer', self.freqs_cos, persistent=False)
        self.register_buffer('freqs_sin_buffer', self.freqs_sin, persistent=False)

    def forward(self, tokens, kv_caches=None, use_cache=False):
        B, Seq = tokens.shape
        h = self.embed_tokens(tokens)

        if kv_caches is not None and kv_caches[0] is not None:
            past_length = kv_caches[0][0].shape[1]
            cos = self.freqs_cos_buffer[past_length:past_length + Seq]
            sin = self.freqs_sin_buffer[past_length:past_length + Seq]
            total_len = past_length + Seq
            mask = torch.full((Seq, total_len), 0.0, device=tokens.device)
        else:
            cos = self.freqs_cos_buffer[:Seq]
            sin = self.freqs_sin_buffer[:Seq]
            mask = torch.full((Seq, Seq), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        new_caches = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(h, cos, sin, mask, layer_cache)
            if use_cache:
                new_caches.append(new_cache)

        h = self.norm(h)
        logits = self.lm_head(h)
        
        if use_cache:
            return logits, new_caches
        return logits

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, use_cache=True):
    model.eval()
    
    inputs = tokenizer.encode(prompt, bos=True, eos=False)
    
    device = torch.cuda.current_device()
    input_ids = torch.tensor(inputs).unsqueeze(0).to(device)

    generated_ids = input_ids.clone()
    
    if is_master():
        print(f"\nGenerating response to: '{prompt}'")
        print("Generated text: ")
        print(prompt, end='', flush=True)

    kv_caches = None
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            if use_cache:
                if i == 0:
                    logits, kv_caches = model(generated_ids, kv_caches=None, use_cache=True)
                else:
                    logits, kv_caches = model(generated_ids[:, -1:], kv_caches=kv_caches, use_cache=True)
            else:
                logits = model(generated_ids)
                
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token.item() in tokenizer.stop_tokens:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if is_master():
                token_text = tokenizer.decode([next_token.item()])
                print(token_text, end="", flush=True)
    
    if is_master():
        print()
        return tokenizer.decode(generated_ids[0].tolist())
    return None

# --- Main ---

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
        print("Weights loaded. Wrapping with FSDP...")


    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )



    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,  # Gradients communication
        buffer_dtype=torch.bfloat16,
    )

    # 4. Wrap the Model
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        forward_prefetch=True,
        )

    if is_master():
        print(f"Model wrapped with FSDP. Running on {dist.get_world_size()} GPUs.")
        ###########
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 0 Memory after wrapping: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
        ###########

    # Initialize Tokenizer
    if Tokenizer:
        tokenizer = Tokenizer(Path(TOKENIZER_MODEL))
    else:
        print("Using dummy tokenizer for test structure.")
        tokenizer = None

    if tokenizer:
        if is_master():
            print("\n" + "=" * 60)
            print("STARTING FSDP GENERATION")
            print("=" * 60)
        
        prompt = "The future of AI is"
        generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0, use_cache=True)
    
            ###########
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 0 Memory after generation: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
        ###########

    if is_master():
        print("\n" + "=" * 60)
        print("STARTING FSDP TRAINING STEP (Dummy)")
        print("=" * 60)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after optimizer turned on: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########

    # Create dummy input batch (Batch Size 4, Sequence Length 128)
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)


    start_train = time.time()
    logits = model(dummy_input)

    ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after forward path : {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = dummy_input[..., 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
    ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after calculated loss: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########
    
    if is_master():
        print(f"Loss computed: {loss.item()}")
        print("Running Backward pass...")

    loss.backward()
    ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after back propagation: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########
    
    if is_master():
        print("Running Optimizer Step...")

    # Optimizer Step
    optimizer.step()
        ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after optimizer step: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########
    optimizer.zero_grad()

    ###########
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU 0 Memory after zero_grad: {allocated:.2f} GB allocated ({reserved:.2f} GB reserved)")
    ###########

    
    if is_master():
        print(f"Training step complete. Time: {time.time() - start_train:.4f}s")
        print("=" * 60)

    cleanup()

if __name__ == "__main__":
    main()