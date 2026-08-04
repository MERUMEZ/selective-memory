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
 ENTORHINAL.PY — Единственные ворота в хранилища
================================================================================
В мозге всё, что входит в гиппокамп и выходит из него, проходит через
энторинальную кору. Не потому, что так удобнее, а потому что кто-то должен
решать, ЧТО КУДА идёт: свежий эпизод — в гиппокамп, устоявшееся знание — в
кору, и обратно тем же путём.

У нас такого места не было. Каждый участок ходил в базу сам: пятьдесят
семь обращений из шести модулей, и ни одно не спрашивало, к какому
хранилищу относится узел.

ЭТО УЖЕ СТОИЛО ЖИВОГО ДЕФЕКТА. Словарь лежит в той же таблице, что и
воспоминания, и подсчёт узлов включал лексику: порог сна пробивался на
девятом сообщении, сон запускался на каждое следующее, кратковременная
память очищалась ежесообщение, а в проде уходило по два вызова языковой
модели на реплику. Починили тогда заплаткой в count_memory_nodes — то
есть отдельным методом, который помнит про типы. Ворота делают такую
ошибку невозможной по устройству: спросить «сколько у меня воспоминаний»
и получить в ответ слоги больше нельзя.

ЧТО ЗДЕСЬ ЕСТЬ:

    gateway.episodic   эпизоды — то, что случилось однажды
    gateway.semantic   слова, слоги, понятия, схемы — что устоялось
    gateway.edges      связи между чем угодно
    gateway.raw        прямой доступ к Database

RAW — ВРЕМЕННЫЙ И НАЗВАН ЧЕСТНО. Пятьдесят семь обращений не переписать
одним заходом без риска, поэтому переход идёт по частям: сначала ворота
существуют и обслуживают то, где тип принципиален, остальное продолжает
ходить напрямую. Каждое обращение через raw — это долг, а не решение.

ЧЕГО ВОРОТА ПОКА НЕ ДЕЛАЮТ. Оба хранилища сидят в одной таблице `nodes`
с колонкой node_type. Разделение на две таблицы — следующий шаг, и ворота
для него и строятся: когда единственное место, знающее про типы, — это
они, перенос семантики в отдельную таблицу не трогает ни один участок.
================================================================================
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Что считается эпизодом, а что — устоявшимся знанием.
#
# Граница проведена по БИОЛОГИИ, а не по удобству: эпизод — то, что
# случилось однажды и имеет время; семантика — то, что накопилось из
# многих случаев и времени не имеет. Свёртка эпизода (episode_summary)
# отнесена к семантике намеренно: она уже не событие, а обобщение.
EPISODIC_TYPES = frozenset({"episodic"})
SEMANTIC_TYPES = frozenset({"word", "syllable", "concept", "episode_summary"})


class EpisodicStore:
    """Эпизоды: то, что случилось однажды. Аналог гиппокампа."""

    def __init__(self, db):
        self._db = db

    def count(self) -> int:
        """
        Сколько ВОСПОМИНАНИЙ, без словаря и без свёрток.

        Отдельный метод существует потому, что ответ на этот вопрос уже
        однажды включал слоги, и сон от этого запускался на каждое
        сообщение.
        """
        return self._db.count_nodes_by_type("episodic")

    def top(self, limit: int) -> List[Any]:
        return self._db.get_top_nodes_by_type("episodic", limit)

    def sample(self, limit: int) -> List[Any]:
        return self._db.get_random_nodes_by_type("episodic", limit)


class SemanticStore:
    """Слова, слоги, понятия, схемы. Аналог коры."""

    def __init__(self, db):
        self._db = db

    def count_words(self) -> int:
        return self._db.count_nodes_by_type("word")

    def count_syllables(self) -> int:
        return self._db.count_nodes_by_type("syllable")

    def count_concepts(self) -> int:
        return self._db.count_nodes_by_type("concept")

    def count_schemas(self) -> int:
        return self._db.count_nodes_by_type("episode_summary")

    def count(self) -> int:
        """Всё корковое вместе."""
        return sum(self._db.count_nodes_by_type(t) for t in sorted(SEMANTIC_TYPES))

    def top_words(self, limit: int) -> List[Any]:
        return self._db.get_top_nodes_by_type("word", limit)


class EdgeStore:
    """Связи. Принадлежат не участку, а самой сети."""

    def __init__(self, db):
        self._db = db

    def all(self) -> List[Any]:
        return self._db.fetch_all_edges()

    def for_node(self, node_id: int) -> List[Any]:
        return self._db.get_edges_for_node(node_id)

    def between(self, node_ids: List[int]) -> List[Any]:
        return self._db.get_edges_between(node_ids)


class Gateway:
    """
    Единственный вход в хранилища и выход из них.

    Держит Database и раздаёт типизированные виды на неё. Прямой доступ
    остаётся под именем `raw` — намеренно неудобным, чтобы каждое такое
    обращение было видно в коде как незакрытый долг.
    """

    def __init__(self, db):
        self.raw = db
        self.episodic = EpisodicStore(db)
        self.semantic = SemanticStore(db)
        self.edges = EdgeStore(db)

    def census(self) -> Dict[str, int]:
        """
        Перепись по хранилищам — для отчётов и для стендов.

        Существует, чтобы вопрос «что вообще внутри» имел ответ, в котором
        воспоминания и словарь названы порознь.
        """
        return {
            "episodes": self.episodic.count(),
            "words": self.semantic.count_words(),
            "syllables": self.semantic.count_syllables(),
            "concepts": self.semantic.count_concepts(),
            "schemas": self.semantic.count_schemas(),
        }

    def close(self) -> None:
        self.raw.close()
