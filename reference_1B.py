import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from safetensors.torch import load_file
from tokenizer import Tokenizer
from pathlib import Path

# Defined Paths
# MODEL_DIR = "/Users/user895/GIT/Llama-3.2-1B/"
# WEIGHTS_PATH = "/Users/user895/GIT/Llama-3.2-1B/model.safetensors"
# CONFIG_PATH = "/Users/user895/GIT/Llama-3.2-1B/config.json"
# TOKENIZER_JSON = "/Users/user895/GIT/Llama-3.2-1B/tokenizer.json"
# TOKENIZER_MODEL = "/Users/user895/GIT/Llama-3.2-1B/original/tokenizer.model"

MODEL_DIR = "/root/Llama-3.2-1B/"
WEIGHTS_PATH = "/root/Llama-3.2-1B/model.safetensors"
CONFIG_PATH = "/root/Llama-3.2-1B/config.json"
TOKENIZER_JSON = "/root/Llama-3.2-1B/tokenizer.json"
TOKENIZER_MODEL = "/root/Llama-3.2-1B/original/tokenizer.model"


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    """
    # Base vocabulary
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class LlamaTokenizerLegacy: # Fast version, not suitable for loss computation
    def __init__(self, json_path):
        print(f"   -> Parsing Tokenizer JSON from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.vocab = self.data["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Map from chars to bytes
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bos_token_id = self.vocab.get("<|begin_of_text|>", 128000)
        self.eos_token_id = self.vocab.get("<|end_of_text|>", 128001)
        self.stop_tokens = [self.eos_token_id, self.vocab.get("<|eot_id|>", 128009)]

    def encode(self, text, add_bos=True):
        """
        Greedy encoder.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        words = text.split(" ")

        for i, word in enumerate(words):
            # 'Ġ' maps to space
            prefix = "Ġ" if i > 0 else ""
            candidate = prefix + word

            if candidate in self.vocab:
                tokens.append(self.vocab[candidate])
            elif word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character matching as failsafe
                for char in candidate:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        # Skip unknown characters since dealing with simple English texts here
                        pass

        return {"input_ids": torch.tensor([tokens], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode ByteLevel BPE tokens.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        text_accumulated = ""
        for tid in token_ids:
            if skip_special_tokens and (
                tid == self.bos_token_id or tid in self.stop_tokens
            ):
                continue
            # Get the raw token string
            text_accumulated += self.id_to_token.get(tid, "")

        # Convert to raw bytes
        text_bytes = bytearray()
        for char in text_accumulated:
            if char in self.byte_decoder:
                text_bytes.append(self.byte_decoder[char])
            else:
                # If char not in the map
                text_bytes.extend(char.encode("utf-8"))

        # Decode raw bytes to utf-8
        decoded_string = text_bytes.decode("utf-8", errors="replace")

        return decoded_string
    


# Implementation copied from Llama repo
# class LlamaTokenizer: -> copied from llama-models repo a separate file


# RoPE


def precompute_freqs_cis(dim: int, end: int, theta: float, scaling_config: dict = None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if scaling_config and scaling_config.get("rope_type") == "llama3":
        factor = scaling_config.get("factor", 32.0)
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        original_max_pos = scaling_config.get("original_max_position_embeddings", 8192)

        # low_freq_wavelen = original_max_pos / low_freq_factor
        # high_freq_wavelen = original_max_pos / high_freq_factor

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
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


# Define model


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

        self.q_proj = nn.Linear(
            config.dim, config.n_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.dim, config.n_kv_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.dim, config.n_kv_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim, config.dim, bias=False
        )

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
            xk = (
                xk.unsqueeze(3)
                .expand(B, Seq_kv, self.n_kv_heads, self.n_rep, self.head_dim)
                .reshape(B, Seq_kv, self.n_heads, self.head_dim)
            )
            xv = (
                xv.unsqueeze(3)
                .expand(B, Seq_kv, self.n_kv_heads, self.n_rep, self.head_dim)
                .reshape(B, Seq_kv, self.n_heads, self.head_dim)
            )

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
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )

        # Move RoPE together with the model
        self.register_buffer('freqs_cos_buffer', self.freqs_cos, persistent = False)
        self.register_buffer('freqs_sin_buffer', self.freqs_sin, persistent = False)

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
            # No cache, process full sequence
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


# Temporary ineffective generation function


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, use_cache=True):
    model.eval()
    inputs = tokenizer.encode(prompt, bos = True, eos = False)
    device = next(model.parameters()).device
    input_ids = torch.tensor(inputs).unsqueeze(0)
    input_ids = input_ids.to(device)

    generated_ids = input_ids.clone()
    print(f"\nGenerating response to: '{prompt}'")
    print("Generated text: ")
    print(prompt, end = '', flush = True)

    kv_caches = None
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            if use_cache:
                if i == 0:
                    # Process full prompt on the first iteration
                    logits, kv_caches = model(generated_ids, kv_caches=None, use_cache=True)
                else:
                    # Process only a single token
                    logits, kv_caches = model(generated_ids[:, -1:], kv_caches=kv_caches, use_cache=True)
            else:
                # Process entire sequence when training
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
            
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)
    
    print()
    return tokenizer.decode(generated_ids[0].tolist())


# Main and Test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(f"Loading Config from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
    config = LlamaConfig(config_data)

    print("Initializing the Model...")
    model = LlamaModel(config)

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
    print("   Model loaded successfully.")

    model = model.to(device)
    print(f'Moved to {device}')

    # print(f"Initializing Manual Tokenizer from {TOKENIZER_JSON}...")
    # tokenizer = LlamaTokenizerLegacy(TOKENIZER_JSON)

    # Initialize meta's implementation of the tokenizer
    tokenizer = Tokenizer(Path(TOKENIZER_MODEL))

    # Generation test
    prompt = "The future of AI is"
    
    # Test with KV-cache
    import time
    print("\n" + "=" * 60)
    print("TESTING WITH KV-CACHE (use_cache=True)")
    print("=" * 60)
    start_time = time.time()
    result_cached = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0, use_cache=True)
    time_cached = time.time() - start_time
    print("-" * 60)
    print("Output:")
    print(result_cached)
    print("-" * 60)
    print(f"Time taken: {time_cached:.4f} seconds")
    print("=" * 60)
    
    # Test without KV-cache
    print("\n" + "=" * 60)
    print("TESTING WITHOUT KV-CACHE (use_cache=False)")
    print("=" * 60)
    start_time = time.time()
    result_no_cache = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0, use_cache=False)
    time_no_cache = time.time() - start_time
    print("-" * 60)
    print("Output:")
    print(result_no_cache)
    print("-" * 60)
    print(f"Time taken: {time_no_cache:.4f} seconds")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Testing a compiled model")
    print("=" * 60)
    model = torch.compile(model, mode="default", fullgraph=False)
    print('warm-up')   
    _ = generate(model, tokenizer, "Warmup", max_new_tokens=5, temperature=0, use_cache=True)
    _ = generate(model, tokenizer, "Warmup", max_new_tokens=5, temperature=0, use_cache=False) 
    start_time = time.time()
    result_compiled = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0, use_cache=True)
    time_compiled = time.time() - start_time
    print("-" * 60)
    print("Output:")
    print(result_compiled)
    print("-" * 60)
    print(f"Time taken: {time_compiled:.4f} seconds")
    print("=" * 60)
    start_time = time.time()
    result_compiled = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0, use_cache=True)
    time_compiled = time.time() - start_time
    print("-" * 60)
    print("Output:")
    print(result_compiled)
    print("-" * 60)
    print(f"Time taken: {time_compiled:.4f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
