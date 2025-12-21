# %% [markdown]
#  # DEFINE VARIABLES

# %%
MODEL_DIR = "/root/Llama-3.2-1B/"
WEIGHTS_PATH = "/root/Llama-3.2-1B/model.safetensors"
CONFIG_PATH = "/root/Llama-3.2-1B/config.json"
TOKENIZER_MODEL = "/root/Llama-3.2-1B/original/tokenizer.model"


# %% [markdown]
#  # IMPORTS

# %%
import re
from datasets import load_dataset
import torch
import torch.nn as nn
from pathlib import Path
from reference_1B import LlamaModel, LlamaConfig
from tokenizer import Tokenizer
import json
from safetensors.torch import load_file
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time



# %% [markdown]
#  # LOAD DATASET

# %%
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

filtered_dataset = []
for entry in dataset['train']['text']:
    if len(entry) > 30:
        filtered_dataset.append(entry)


# %%
def clean_wikitext(text):
    text = text.replace(' @-@ ', '-')
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+(\'[a-zA-Z])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    text = text.strip()
    
    return text


# %%
wiki_dataset = [clean_wikitext(entry) for entry in filtered_dataset]


# %% [markdown]
#  # INITIALIZE THE MODEL

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.bfloat16


# %%
# Load config and initialize model
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
config = LlamaConfig(config_data)
model = LlamaModel(config)
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
model = model.to(device, dtype = dtype)


# %%
tokenizer = Tokenizer(Path(TOKENIZER_MODEL))


# %%
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3


# %% [markdown]
#  # DEFINITION

# %%
def tokenize_batch(texts):
    tokenized_texts = []
    for entry in texts:
        tokens = tokenizer.encode(entry, bos=True, eos=True)
        tokenized_texts.append(tokens)

    max_length = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = []
    loss_masks = []
    pad_token_id = 128001

    for tokens in tokenized_texts:
        pad_length = max_length - len(tokens)
        padded_tokens = tokens + [pad_token_id] * pad_length
        loss_mask = [1] * len(tokens) + [0] * pad_length
        padded_texts.append(padded_tokens)
        loss_masks.append(loss_mask)

    input_ids = torch.tensor(padded_texts, dtype=torch.long).to(device)
    loss_masks = torch.tensor(loss_masks, dtype=torch.bool).to(device)

    return input_ids, loss_masks


# %% [markdown]
#  # Train on a chunk without repetitions

# %%
class WikiTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        return text[0:500] #500 to avoid long outliers causing OOM


# %%
train_dataset = WikiTextDataset(wiki_dataset[1000:3000])


# %%
def train_epoch(batch_size = 2): # not encaplusated

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Randomized order
)


    scaler = GradScaler()
    losses = []


    start_time = time.time()
    print(start_time)
    for i, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
        input_ids, loss_masks = tokenize_batch(batch)

        optimizer.zero_grad()   
        with torch.autocast('cuda'):
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Calculate loss
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass
            loss.backward()
            losses.append(loss.detach().cpu())
        optimizer.step()
    end_time = time.time()
    print(end_time)
    print(start_time - end_time)
    
    return losses

# %% [markdown]
# # TRAIN WITH LORA
# 

# %%
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
)
config.get = lambda key, default=None: getattr(config, key, default)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.to(device, dtype=dtype)

optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)

# %%



