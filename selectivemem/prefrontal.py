# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# A commercial licence is available for use in closed products — see
# COMMERCIAL.md.
"""
================================================================================
 PREFRONTAL.PY — Рабочая память: что удерживается прямо сейчас
================================================================================
WorkingMemory is a buffer of the last N turns (role, text, emotion_score,
perplexity, timestamp) — working memory in the psychological sense: it
holds the thread of the current conversation here and now, regardless of
whether any of it ever reaches long-term storage.

STM does NOT decide what matters — GraphMemory.consolidate_from_stm()
does (selective consolidation). WorkingMemory is a plain bounded buffer
(deque) with no decision logic of its own.
================================================================================
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from selectivemem.settings import MemorySettings


@dataclass
class STMEntry:
    """A single entry in short-term memory."""
    role: str          # "user" | "bot"
    text: str
    emotion_score: float = 0.0
    perplexity: float = 0.0
    timestamp: Optional[float] = None


class WorkingMemory:
    """
    Short-term memory buffer built on collections.deque.

    Usage:
        stm = WorkingMemory()
        stm.add_message("user", "hello!", emotion_score=0.2, perplexity=0.3)
        stm.add_message("bot", "hi, how are you?")

        if stm.is_full():
            episode = stm.consume_all()  # take everything and clear the buffer
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        settings: Optional[MemorySettings] = None,
    ):
        self.settings = settings or MemorySettings()
        self.capacity = capacity if capacity is not None else self.settings.stm_capacity
        self._buffer: Deque[STMEntry] = deque(maxlen=self.capacity)

    def add_message(
        self,
        role: str,
        text: str,
        emotion_score: float = 0.0,
        perplexity: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Appends a turn to the buffer. Once it is full, deque evicts the
        oldest entry automatically (FIFO).
        """
        entry = STMEntry(
            role=role,
            text=text,
            emotion_score=emotion_score,
            perplexity=perplexity,
            timestamp=timestamp,
        )
        self._buffer.append(entry)

    def get_context_string(self) -> str:
        """
        The current conversation as a readable string, ready to be mixed
        into a system prompt.

        Format:
            User: hello!
            Bot: hi, how are you?
            User: tell me about yourself
        """
        if not self._buffer:
            return ""

        lines = []
        for entry in self._buffer:
            speaker = "User" if entry.role == "user" else "Bot"
            lines.append(f"{speaker}: {entry.text.strip()}")

        return "\n".join(lines)

    def is_full(self) -> bool:
        """True when the buffer is full."""
        return len(self._buffer) >= self.capacity

    def size(self) -> int:
        """How many entries the buffer currently holds."""
        return len(self._buffer)

    def get_status_string(self) -> str:
        """A string like '4/6 items in buffer' for debug output."""
        return f"{len(self._buffer)}/{self.capacity} items in buffer"

    def get_entries(self) -> List[STMEntry]:
        """A copy of the current entries, leaving the buffer intact."""
        return list(self._buffer)

    def consume_all(self) -> List[STMEntry]:
        """
        Takes ALL current entries and empties the buffer. Called before
        consolidation into long-term storage so the same turns are never
        processed twice.
        """
        entries = list(self._buffer)
        self._buffer.clear()
        return entries

    def clear(self) -> None:
        """Empties the buffer without returning its contents."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={len(self._buffer)}/{self.capacity})"