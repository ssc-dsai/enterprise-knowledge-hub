"""Tokenizer module"""
from dataclasses import dataclass
from typing import Any
import threading

@dataclass
class ThreadTokenizer:
    """
    Lazily creates one tokenizer per thread for a given model_name.
    This avoids locks and avoids sharing a single tokenizer instance across threads.
    """
    model_name: str
    use_fast: bool = True

    _local: threading.local = threading.local()

    def get(self) -> Any:
        """Getter for tokenizer from model"""
        tok = getattr(self._local, "tokenizer", None)
        if tok is None:
            from transformers import AutoTokenizer # pylint: disable=import-outside-toplevel

            tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast)
            self._local.tokenizer = tok
        return tok

    def is_loaded_for_current_thread(self) -> bool:
        """Checks if tokenizer has been loaded for thread"""
        return getattr(self._local, "tokenizer", None) is not None
