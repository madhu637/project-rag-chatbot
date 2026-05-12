"""Sliding-window conversation memory (no LangChain dependency)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class Turn:
    question: str
    answer: str


class ConversationBufferMemory:
    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self._buffer: Deque[Turn] = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        self._buffer.append(Turn(question=question, answer=answer))

    def to_prompt(self) -> str:
        if not self._buffer:
            return ""
        lines: List[str] = []
        for turn in self._buffer:
            lines.append(f"User: {turn.question}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def turns(self) -> List[Turn]:
        return list(self._buffer)
