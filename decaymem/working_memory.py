# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# Эта программа — свободное ПО: вы можете распространять и изменять её
# на условиях GNU Affero General Public License версии 3, изданной
# Free Software Foundation. Полный текст — в файле LICENSE.
#
# Программа распространяется В НАДЕЖДЕ, ЧТО БУДЕТ ПОЛЕЗНОЙ, но БЕЗ
# ВСЯКИХ ГАРАНТИЙ, включая подразумеваемые гарантии товарного
# состояния и пригодности для определённой цели.
#
# Для использования в закрытых продуктах существует коммерческая
# лицензия — см. COMMERCIAL.md.
"""
================================================================================
 WORKING_MEMORY.PY — Кратковременная память (STM) "Динамического Мозга"
================================================================================
WorkingMemory — буфер последних N реплик диалога (роль, текст, emotion_score,
perplexity, timestamp), реализующий "рабочую память": удержание нити текущего
разговора "здесь и сейчас", независимо от того, попадёт ли что-то из этого
в долгосрочную память (LTM/GraphMemory).

STM НЕ решает, что важно — это делает GraphMemory.consolidate_from_stm()
(Избирательная Консолидация). WorkingMemory — чистый буфер с ограниченной
ёмкостью (deque), без "биологической" логики принятия решений.
================================================================================
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from decaymem.settings import MemorySettings


@dataclass
class STMEntry:
    """Одна запись в кратковременной памяти."""
    role: str          # "user" | "bot"
    text: str
    emotion_score: float = 0.0
    perplexity: float = 0.0
    timestamp: Optional[float] = None


class WorkingMemory:
    """
    Буфер кратковременной памяти (STM) на основе collections.deque.

    Использование:
        stm = WorkingMemory()
        stm.add_message("user", "привет!", emotion_score=0.2, perplexity=0.3)
        stm.add_message("bot", "привет, как дела?")

        if stm.is_full():
            episode = stm.consume_all()  # забрать всё и очистить буфер
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
        Добавляет новую реплику в буфер STM. Если буфер уже заполнен,
        deque автоматически вытесняет самую старую запись (FIFO).
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
        Возвращает срезовый контекст текущего диалога в виде читаемой
        строки, готовой к подмешиванию в system-промпт LLM.

        Формат:
            User: привет!
            Bot: привет, как дела?
            User: расскажи о себе
        """
        if not self._buffer:
            return ""

        lines = []
        for entry in self._buffer:
            speaker = "User" if entry.role == "user" else "Bot"
            lines.append(f"{speaker}: {entry.text.strip()}")

        return "\n".join(lines)

    def is_full(self) -> bool:
        """Возвращает True, если буфер STM заполнен до отказа."""
        return len(self._buffer) >= self.capacity

    def size(self) -> int:
        """Текущее количество элементов в буфере."""
        return len(self._buffer)

    def get_status_string(self) -> str:
        """Строка вида '4/6 items in buffer' — для [BRAIN DEBUG]."""
        return f"{len(self._buffer)}/{self.capacity} items in buffer"

    def get_entries(self) -> List[STMEntry]:
        """Возвращает копию текущих записей без очистки буфера."""
        return list(self._buffer)

    def consume_all(self) -> List[STMEntry]:
        """
        Забирает ВСЕ текущие записи из STM и полностью очищает буфер.
        Используется перед консолидацией в LTM (GraphMemory), чтобы
        избежать повторной обработки одних и тех же реплик.
        """
        entries = list(self._buffer)
        self._buffer.clear()
        return entries

    def clear(self) -> None:
        """Полностью очищает буфер без возврата содержимого."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={len(self._buffer)}/{self.capacity})"