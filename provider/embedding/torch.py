import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from provider.embedding.base import EmbeddingBackendProvider


class TorchEmbeddingBackend(EmbeddingBackendProvider):
    def __init__(self, model_name: str, device: str = "cuda", max_seq_len: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_seq_len = max_seq_len
        
        #for testing purposes here
        local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True)
        self.model = AutoModel.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf", local_files_only=True).to(device)
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(device)
        
        self.model.eval()
        
    def setDevice(self, device: str):
        self.device = device        

    def embed_torch(self, texts: List[str]) -> np.ndarray:
        
        #this means i'm retokenizing i think?  doing it twice.  
        #chunk token by text, tokenizes, then decodes back to text. 
            # maybe don't decode.  
        # check how long it takes, we can deal with this when we split embed and tokenizing to ensure optimizing by time. 
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # simple mean pooling
        # change this to be dynamic based on modeL?
        last_hidden = outputs.last_hidden_state      # [B, T, D]
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1)                    # [B, 1]
        embeddings = summed / lengths

        return embeddings.cpu().numpy()