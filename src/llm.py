"""Thin wrapper around the Hugging Face Inference API."""
from __future__ import annotations

import os
from typing import Optional

from huggingface_hub import InferenceClient


SYSTEM_PROMPT = (
    "You are a precise document-grounded assistant. Answer the user's question "
    "using ONLY the information in the provided context. If the answer is not in "
    "the context, reply exactly: 'I could not find that in the provided documents.' "
    "Always cite the source as (filename, page N). Be concise."
)


class HFChatLLM:
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.client = InferenceClient(model=model, token=token)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def chat(self, context: str, history: str, question: str) -> str:
        """Run a single chat completion grounded on the provided context."""
        user_block = (
            f"Conversation so far:\n{history or '(none)'}\n\n"
            f"Context from documents:\n{context}\n\n"
            f"Question: {question}"
        )
        try:
            resp = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_block},
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            # Fallback: try text-generation endpoint (some models don't support chat)
            prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_block} [/INST]"
            try:
                out = self.client.text_generation(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    return_full_text=False,
                )
                return out.strip()
            except Exception:
                raise exc
