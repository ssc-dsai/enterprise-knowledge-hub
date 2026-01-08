"""Utility helpers for embedding workflows."""

# pylint: disable=invalid-name

import gc
import os
import time
from typing import Callable, Dict, Iterable, Tuple

import torch
from llama_cpp import Llama  # pylint: disable=no-name-in-module
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

class EmbeddingUtil:
    """Embedding helper utilities for chunking and batch sizing."""

    @staticmethod
    def chunk_text_by_tokens(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> list[str]:
        """Split text into overlapping chunks based on tokenizer limits."""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start = end - overlap_tokens  # move forward but keep an overlap

        return chunks

    @staticmethod
    def article_to_chunks(article: Dict, tokenizer: PreTrainedTokenizerBase,
                          max_tokens: int = 512, overlap_tokens: int = 64):
        """Yield chunked article content with metadata for storage."""
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
    def batched(iterable: Iterable[Dict], batch_size: int):
        """Yield items from iterable in batches of batch_size."""
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
        max_seq_len: int = 512,
        device: str = "cuda",
        start_batch: int = 1,
        max_batch_cap: int = 4096,
    ) -> int:
        """Binary search the largest batch size that fits in memory for a torch model."""
        if max_batch_cap == 4096 and device == "cpu":
            max_batch_cap = 1024

        local_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/"
            "snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            local_dir,
            gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
            local_files_only=True,
        )

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
            except RuntimeError as exc:
                print(f"runtime======{str(exc).lower()}")
                if "out of memory" in str(exc).lower():
                    print(f"OOM at batch_size={batch_size}")
                    return False
                raise
            return True

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

        if high > max_batch_cap:
            print(f"Reached max_batch_cap={max_batch_cap} without OOM.")
            return max_batch_cap

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

    @staticmethod
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def detect_max_batch_size_llamacpp(
        model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int | None = None,
        pooling_type: int = 0,
        start_n_batch: int = 128,
        max_n_batch_cap: int = 8192,
        ubatch_policy: Callable[[int], int] = lambda nb: min(nb, 512),
        probe_texts_per_call: int = 8,
        probe_text_chars: int = 6000,
        normalize: bool = True,
        warmup: bool = True,
        verbose: bool = True,
    ) -> Tuple[int, int]:
        """Determine the largest llama.cpp n_batch and n_ubatch that fit memory."""

        dummy_text = "x" * probe_text_chars
        probe_inputs = [dummy_text] * probe_texts_per_call

        def try_run(n_batch: int) -> bool:
            n_ubatch = ubatch_policy(n_batch)

            try:
                llm = Llama(
                    model_path=model_path,
                    embedding=True,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    n_batch=n_batch,
                    n_ubatch=n_ubatch,
                    pooling_type=pooling_type,
                    verbose=False,
                )

                if warmup:
                    _ = llm.embed([probe_inputs[0]], normalize=normalize, truncate=True)

                _ = llm.embed(probe_inputs, normalize=normalize, truncate=True)
                return True

            except Exception as exc:  # pylint: disable=broad-except
                if verbose:
                    print(f"[FAIL] n_batch={n_batch}, n_ubatch={n_ubatch} -> {type(exc).__name__}: {exc}")
                return False

            finally:
                try:
                    del llm
                except UnboundLocalError:
                    pass
                gc.collect()
                time.sleep(0.05)

        lo = start_n_batch
        if not try_run(lo):
            return 0, 0

        hi = lo
        while True:
            next_hi = min(hi * 2, max_n_batch_cap)

            if next_hi == hi:
                break

            if try_run(next_hi):
                lo = next_hi
                hi = next_hi
                if hi >= max_n_batch_cap:
                    break
            else:
                hi = next_hi
                break

        if lo == max_n_batch_cap:
            return lo, ubatch_policy(lo)

        left = lo
        right = hi
        best = lo

        while left + 1 < right:
            mid = (left + right) // 2
            if try_run(mid):
                best = mid
                left = mid
            else:
                right = mid

        return best, ubatch_policy(best)
