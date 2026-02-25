import tiktoken
from typing import Any

class TextChunker:

    max_seq_length: int
    tt: Any
    
    def __init__(self):
        self.tt = tiktoken.get_encoding("cl100k_base")
        
    def chunk_text_by_tokens(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int = 10
    ) -> list[str]:
        """Split text into chunks based on token count with overlap using tiktoken."""

        # Encode entire text
        tokens = self.tt.encode(text)

        # If text fits in one chunk
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.tt.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move forward with overlap
            start_idx += max_tokens - overlap_tokens

        return chunks