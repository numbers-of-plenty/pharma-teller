# %%
import json

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

INPUT_FILE = "input.jsonl"  # TODO replace with actual later
OUTPUT_FILE = "embeddings_output.jsonl"
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
TARGET_DIM = 2048
BATCH_SIZE = 2


class NewsDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading {MODEL_NAME} on {device}...")

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    model.truncate_dim = TARGET_DIM
    dataset = NewsDataset(INPUT_FILE)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    with open(OUTPUT_FILE, "w") as f_out:
        for batch_num, batch in tqdm(enumerate(dataloader, 1)):
            titles = [item["title"] for item in batch]
            embeddings = model.encode(
                titles, convert_to_numpy=False, normalize_embeddings=True
            )

            for item, embedding in zip(batch, embeddings):
                item["embedding"] = embedding.cpu().tolist()
                f_out.write(json.dumps(item) + "\n")

    print(f"\nComplete! Processed {batch_num} batches. Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


# %%

client = chromadb.PersistentClient(path="./chroma_db_embedded_news")
collection = client.get_or_create_collection(
    name="news_embeddings", metadata={"description": "News articles with embeddings"}
)

documents = []
embeddings = []
metadatas = []
ids = []

with open(OUTPUT_FILE, "r") as f:
    for idx, line in tqdm(enumerate(f)):
        item = json.loads(line)

        documents.append(item["title"])
        embeddings.append(item["embedding"])
        metadatas.append(
            {"date": item["date"], "days_since_2000": item["days_since_2000"]}
        )
        ids.append(str(idx))

# Batch adding
CHROMA_BATCH_SIZE = 5000
for i in tqdm(range(0, len(documents), CHROMA_BATCH_SIZE)):
    end_idx = min(i + CHROMA_BATCH_SIZE, len(documents))

    collection.add(
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx],
        metadatas=metadatas[i:end_idx],
        ids=ids[i:end_idx],
    )


# %%
