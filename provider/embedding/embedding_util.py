"""
Utility class
"""
from typing import Dict, Iterable
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModel
import torch

class EmbeddingUtil:
    """
    Provides utility functions for embedding
    """
    @staticmethod
    def chunk_text_by_tokens(text: str,
                             tokenizer: PreTrainedTokenizerBase, max_tokens: int = 512, overlap_tokens: int = 64):
        """
        This module provides functions for calculating basic arithmetic operations.
        """
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            # move forward but keep an overlap
            start = end - overlap_tokens

        return chunks

    #make this abstract?  the yield is wiki specific
    @staticmethod
    def article_to_chunks(article: Dict, tokenizer, max_tokens=512, overlap_tokens=64):
        """
        Chunks articles
        """
        chunks = EmbeddingUtil.chunk_text_by_tokens(article["xml_content"], tokenizer, max_tokens, overlap_tokens)
        for i, chunk_text in enumerate(chunks):
            yield {
                "id": f'{article["page_id"]}-{i}',
                "text": chunk_text,
                "metadata": {
                    "article_id": article["page_id"],
                    "title": article.get("title"),
                    "chunk_index": i,
                },
            }

    @staticmethod
    def batched (iterable: Iterable[Dict], batch_size: int):
        """
        Write meaningful stuff
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def detect_max_batch_size_torch(
        model_name: str,
        max_seq_len: int = 512,
        device: str = "cuda",
        start_batch: int = 1,
        max_batch_cap: int = 4096,
    ):
        """
        Embedding Provider base
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # if default and device is cpu, then lower batch cap
        if max_batch_cap == 4096 and device == 'cpu':
            max_batch_cap = 1024

        # for testing lcoal
        # local_dir = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/
        # 370f27d7550e0def9b39c1f16d3fbaa13aa67728/")
        # tokenizer = AutoTokenizer.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
        #                                           local_files_only=True)
        # model = AutoModel.from_pretrained(local_dir, gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
        # local_files_only=True)

        model.to(device)
        model.eval()

        dummy_text = "This is a dummy sentence for batch size testing."

        def can_run(batch_size: int) -> bool:
            try:
                texts = [dummy_text] * batch_size
                inputs = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_len,
                    return_tensors="pt",
                ).to(device)

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=device)
                with torch.no_grad():
                    _ = model(**inputs)
                return True
            except RuntimeError as e:
                print ('runtime======' + str(e).lower())
                if "out of memory" in str(e).lower():

                    # Clear OOM state
                    print(f"OOM at batch_size={batch_size}")
                    # torch.cuda.empty_cache()
                    return False
                return True

        # Phase 1: exponential search to find an upper bound where OOM happens
        low = 0
        high = start_batch

        while high <= max_batch_cap:
            print(f"Trying batch_size={high}...")
            if can_run(high):
                low = high
                high *= 2
            else:
                break

        if low == 0:
            raise RuntimeError("Even batch_size=1 does not fit in GPU memory.")

        # If we never hit OOM up to max_batch_cap, just return that cap
        if high > max_batch_cap:
            print(f"Reached max_batch_cap={max_batch_cap} without OOM.")
            return max_batch_cap

        # Phase 2: binary search between low (good) and high (bad)
        print(f"Binary search between {low} (ok) and {high} (OOM)")
        while low + 1 < high:
            mid = (low + high) // 2
            print(f"Trying batch_size={mid}...")
            if can_run(mid):
                low = mid
            else:
                high = mid

        print(f"Max safe batch size = {low}")
        return low
