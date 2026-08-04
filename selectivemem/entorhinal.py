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

    gateway.node       действия над узлом по номеру, безразличные к типу
    gateway.raw        прямой доступ к Database

ПЕРЕХОД ЗАВЕРШЁН: прямых обращений к базе в участках НЕТ НИ ОДНОГО.
Пятьдесят семь вызовов переведены, и распределение оказалось
информативнее самого счёта:

    по номеру узла, безразличны к типу   31   после разделения работают как есть
    зависят от типа                      15   вот настоящая цена разделения
    рёбра                                 9   самая трудная часть: пересекают границу
    вся таблица                           2

То есть дорого не «пятьдесят семь мест», а пятнадцать типозависимых плюс
рёбра. Это и есть оценка, ради которой этап делался.

ЧЕГО ВОРОТА ПОКА НЕ ДЕЛАЮТ. Оба хранилища сидят в одной таблице `nodes`
с колонкой node_type. Разделение на две таблицы взвешено и ОТЛОЖЕНО — у
него не нашлось измеримой выгоды (см. ARCHITECTURE.ru.md). Ворота
остаются готовыми: когда единственное место, знающее про типы, — это они,
перенос семантики не тронет ни один участок.
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


class NodeStore:
    """
    Действия над узлом ПО ЕГО НОМЕРУ, безразличные к типу.

    Половина всех обращений к базе — именно такие: прочитать, поднять
    силу, обновить вес, удалить. Им всё равно, в каком хранилище лежит
    узел, и после разделения они будут работать без единой правки.

    Это и есть настоящая оценка цены разделения: дорого не «пятьдесят
    семь обращений», а те пятнадцать, что зависят от типа, плюс рёбра,
    пересекающие границу.
    """

    def __init__(self, db):
        self._db = db

    def get(self, node_id: int):
        return self._db.get_node(node_id)

    def many(self, node_ids: List[int]):
        return self._db.get_nodes_by_ids(node_ids)

    def degrees(self, node_ids: List[int]) -> Dict[int, int]:
        return self._db.get_degrees(node_ids)

    def add_strength(self, node_id: int, delta: float, cap: float) -> None:
        self._db.add_strength(node_id, delta, cap)

    def set_weight(self, node_id: int, weight: float) -> None:
        self._db.update_weight(node_id, weight)

    def set_weights(self, updates: List[Dict[str, Any]]) -> None:
        self._db.bulk_update_weights(updates)

    def set_stability(self, node_id: int, stability: float) -> None:
        self._db.update_stability(node_id, stability)

    def set_embedding(self, node_id: int, blob) -> None:
        self._db.update_embedding(node_id, blob)

    def set_reward_expectation(self, node_id: int, value: float) -> None:
        self._db.update_reward_expectation(node_id, value)

    def touch(self, node_id: int, timestamp: Optional[float] = None) -> None:
        self._db.update_last_accessed(node_id, timestamp=timestamp)

    def delete(self, node_id: int) -> None:
        self._db.delete_node(node_id)

    def all(self):
        """Все узлы без разбора. После разделения станет объединением двух."""
        return self._db.fetch_all_nodes()


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

    def insert(self, **kwargs) -> Optional[int]:
        """Новый эпизод. node_type задаётся здесь, а не вызывающим."""
        kwargs.setdefault("node_type", "episodic")
        return self._db.insert_node(**kwargs)

    def searchable(self) -> List[Any]:
        """
        То, что участвует в поиске: эпизоды и явно заданные понятия.

        Лексика исключена — её узлы служебные, и однажды они давали ложные
        совпадения на любом повторённом слове. Свёртки тоже: они содержат
        столько слов, что совпадают почти с любым запросом, и в общей
        выдаче роняли R@1 с 76% до 42%.
        """
        return self._db.fetch_searchable_nodes()

    def candidates_by_text(self, words: List[str], limit: int) -> List[Any]:
        return self._db.fetch_candidates_by_text(words, limit)

    def document_frequency(self, words: List[str]) -> Dict[str, int]:
        return self._db.document_frequency(words)

    def orphans(self, min_edge_weight: float, max_node_weight: float) -> List[Any]:
        return self._db.get_orphan_nodes(min_edge_weight, max_node_weight)


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

    def sample_syllables(self, limit: int) -> List[Any]:
        return self._db.get_random_nodes_by_type("syllable", limit)

    def count_mastered(self, min_weight: float) -> int:
        return self._db.count_mastered_words(min_weight)

    def lexical_by_texts(self, node_type: str, texts: List[str]) -> List[Any]:
        return self._db.get_lexical_nodes_by_texts(node_type, texts)

    def upsert_lexical(self, *args, **kwargs):
        return self._db.upsert_lexical_node(*args, **kwargs)

    def upsert_concept(self, *args, **kwargs):
        return self._db.upsert_concept_node(*args, **kwargs)

    def meta(self, node_type: str):
        """
        Мета-узел: модель себя, модель собеседника, эпоха мозга.

        Отнесены к семантике, потому что это УСТОЯВШЕЕСЯ знание о
        постоянном, а не событие: «я такой-то» не случилось однажды.
        """
        return self._db.get_meta_node(node_type)

    def upsert_meta(self, *args, **kwargs):
        return self._db.upsert_meta_node(*args, **kwargs)

    def schemas(self, limit: int) -> List[Any]:
        return self._db.fetch_summary_nodes(limit)

    def hub_candidates(self, min_edge_weight: float) -> List[Any]:
        return self._db.get_hub_candidates(min_edge_weight)

    def insert_schema(self, **kwargs) -> Optional[int]:
        kwargs.setdefault("node_type", "episode_summary")
        return self._db.insert_node(**kwargs)


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

    def upsert(self, *args, **kwargs):
        return self._db.upsert_edge(*args, **kwargs)

    def set_weights(self, updates: List[Dict[str, Any]]) -> None:
        self._db.bulk_update_edge_weights(updates)

    def delete(self, edge_id: int) -> None:
        self._db.delete_edge(edge_id)

    def delete_below(self, min_weight: float) -> int:
        return self._db.delete_edges_below_weight(min_weight)


class Gateway:
    """
    Единственный вход в хранилища и выход из них.

    Держит Database и раздаёт типизированные виды на неё. Прямой доступ
    остаётся под именем `raw` — намеренно неудобным, чтобы каждое такое
    обращение было видно в коде как незакрытый долг.
    """

    def __init__(self, db):
        self.raw = db
        self.node = NodeStore(db)
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
