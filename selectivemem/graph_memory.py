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
 GRAPH_MEMORY.PY — Сборка участков в одну память
================================================================================
Здесь почти нет логики. MemoryGraph — место, где участки соединяются и
где лежит общее для них состояние: база, настройки, кодировщик.

    hippocampus.py    быстрая запись эпизода, рассогласование, связывание
    neocortex.py      медленное знание: словарь, понятия, схемы, удивление
    consolidation.py  перенос гиппокамп -> кора во сне
    retrieval.py      извлечение: предотбор, оценка, растекание
    synapses.py       уровень связи: вес, затухание, ёмкость
    records.py        структуры ответа
    stopwords.py      служебные слова
    prefrontal.py     рабочая память: что удерживается прямо сейчас
    plasticity.py     порог записи — модуляция пластичности
    reinforcement.py  дофаминовый контур: ошибка предсказания

РАЗДЕЛЕНИЕ ПО НОСИТЕЛЮ, А НЕ ПО ПРОЦЕССУ, и это второй заход. Первый
разнёс файл по функциям — «лексика», «забывание», «консолидация», — и
получилась инженерная нарезка в биологических словах. Она разводила по
трём файлам словарь, понятия и схемы, хотя это одно и то же вещество:
медленно накапливаемая корковая статистика. И приписывала абстракцию сну,
хотя сон её лишь запускает, а живёт она в коре.

ЧТО ЭТО ДАЛО, КРОМЕ ИМЁН. Разделение по носителю сразу показывает
ПРОБЕЛЫ — то, чему не нашлось владельца:

  * РАЗДЕЛЕНИЯ ОБРАЗОВ НЕТ. Зубчатая извилина делает похожие входы
    непохожими ДО записи, чтобы потом они не мешали друг другу. У нас
    похожее пишется как похожее, и стенд двойников это показывает: R@1
    со 100% до 50% на пятидесяти конкурентах. Мы боремся с последствием
    при извлечении, тогда как биология решает это на входе.
  * ВОРОТ НЕТ. В мозге всё в гиппокамп и из него идёт через энторинальную
    кору. У нас каждый участок ходит в базу сам.
  * МИНДАЛИНЫ НЕТ. Значимость приходит параметром снаружи, по умолчанию
    ноль, то есть у обычного пользователя половина формулы гейта мертва.
  * ХРАНИЛИЩЕ ОДНО. Гиппокамп и кора — разные хранилища с разными
    правилами, а у нас одна таблица nodes с колонкой node_type. Это самое
    глубокое расхождение, и оно уже стоило одного живого дефекта: подсчёт
    узлов включал словарь, и сон запускался на каждое сообщение.

РАЗДЕЛЕНИЕ ПОКА ФАЙЛОВОЕ. Участки — миксины: состояние принадлежит этому
классу. Настоящее владение данными по участкам ломает все вызовы и
делается отдельно, с полным перемером.
================================================================================
"""

import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from selectivemem.database import Database
from selectivemem.settings import MemorySettings
from selectivemem import embeddings
# Структуры ответа живут в records.py — иначе выделенные по участкам
# модули и это ядро импортировали бы друг друга по кругу. Имена
# ре-экспортируются: снаружи их берут отсюда тесты, стенды и витрина.
# Словарь и удивление живут в lexicon.py: это корковое знание, а не
# эпизодическая память, и держать их в одном файле было ошибкой, уже
# стоившей одного живого дефекта — см. шапку lexicon.py.
# Консолидация и сон — в consolidation.py: реактивация, понижение,
# подрезка и свёртка кластера в схему. Это работа, которую память делает,
# когда её не спрашивают, и держать её вперемешку с записью и поиском
# значило не замечать, что двух первых стадий у сна вовсе нет.
from selectivemem.consolidation import ConsolidationMixin
# Ворота: единственный типизированный вход в хранилища. См. entorhinal.py —
# там же записано, почему прямой доступ к базе однажды стоил живого дефекта.
from selectivemem.entorhinal import Gateway
# Восприятие, которое организм отращивает сам, когда своего кодировщика
# не дали. См. perception.py — там же, почему без него не работает
# половина устройства.
from selectivemem.perception import GrownEncoder
from selectivemem.hippocampus import HippocampusMixin
# Забывание — в forgetting.py. Там же записано, что теорий в нём сейчас
# две: ранжирование живёт по интерференции, а само затухание всё ещё
# распад по часам. Переход остановлен на середине сознательно.
from selectivemem.synapses import SynapsesMixin
# Извлечение — в retrieval.py: поиск, предотбор, оценка, растекание.
from selectivemem.retrieval import RetrievalMixin
from selectivemem.neocortex import VOWELS, WORD_PATTERN, NeocortexMixin  # noqa: F401
from selectivemem.stopwords import STOP_WORDS  # noqa: F401
from selectivemem.records import (  # noqa: F401
    ActivationTrace,
    AssociatedNode,
    ConsolidationResult,
    HubCluster,
    KnownSyllable,
    KnownWord,
    LexicalProcessingResult,
    MemoryMatch,
    PruningReport,
    RewardSignal,
    SupersededNode,
    SurpriseResult,
)
import logging

if TYPE_CHECKING:
    from selectivemem.prefrontal import STMEntry

logger = logging.getLogger(__name__)

# Stop words for keyword matching. Russian entries are here because the
# bundled encoder is Russian; English ones because that is the other
# language this was tested in. With your own encoder and language, extend
# this set — it only filters noise from the lexical component of scoring.


# WORD_PATTERN и VOWELS переехали в lexicon.py и импортируются ниже:
# они описывают устройство языка, а не памяти.

# Vowels (Russian + English) for primitive syllable segmentation: a
# syllable is consonants* + vowel(s), and trailing consonants stick to the
# last syllable of the word. Add your own alphabet's vowels here if the
# babbling showcase matters to you; the memory itself does not use this.




class MemoryGraph(
    HippocampusMixin,
    NeocortexMixin,
    ConsolidationMixin,
    RetrievalMixin,
    SynapsesMixin,
):
    """
    The high-level interface to the long-term memory graph.

    Usage:
        graph = MemoryGraph()
        graph.save_connection("hello", "hi, how are you?", weight=0.9)
        matches = graph.search("how are you doing")
        graph.apply_decay()
        result = graph.consolidate_from_stm(stm_entries, timestamp=brain_time)
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        settings: Optional[MemorySettings] = None,
        encoder: Optional[Callable[[str], Any]] = None,
    ):
        # Parameters arrive from outside: the core must not know about
        # the application's global config, or it could never be extracted
        # into a package of its own. The defaults in MemorySettings are the
        # values calibrated by measurement, so MemoryGraph() with no
        # arguments behaves exactly as before.
        self.settings = settings or MemorySettings()
        self.db = db or Database(settings=self.settings)
        # Ворота держат ту же базу и раздают типизированные виды на неё.
        self.gate = Gateway(db)
        # ВОСПРИЯТИЕ. Растёт от каждого сообщения и восстанавливается из
        # сохранённых эпизодов при открытии базы: проснувшись, организм
        # заново выводит смыслы из того, что помнит.
        self.perception: Optional[GrownEncoder] = None
        if self.settings.grow_perception and encoder is None:
            self.perception = GrownEncoder()
            self._relearn_perception()

        # THE MEANING ENCODER IS PLUGGABLE. The bundled one is navec,
        # Russian static vectors: light and free of torch, but Russian.
        # For English — or any other language — pass your own function
        # text -> vector: sentence-transformers, embeddings over an API,
        # fastText, anything that returns a sequence of numbers or None.
        #
        # The library SHIPS NO MODEL and does not choose one for the user.
        # That also settles the question "what language does selectivemem
        # work in": whichever language the supplied encoder works in.
        # Stores EXACTLY what was passed rather than defaulting to
        # embeddings.encode: binding the function here would nail the
        # reference down, and replacing embeddings.encode from outside
        # (tests, benchmarks, disabling semantics on the fly) would stop
        # working. This project already stepped on that rake with the LLM
        # stub.
        self.encoder = encoder
        self._vector_dim: Optional[int] = None
        self._warned_no_semantics = False
        self.last_activation_traces: List[ActivationTrace] = []

    # ----------------------------------------------------------------------
    # SELF-MODEL & USER-MODEL — meta-node initialisation
    # ----------------------------------------------------------------------


    def _relearn_perception(self) -> None:
        """
        Восстановить восприятие из того, что организм помнит.

        Отпечатки слов детерминированы от самих слов, поэтому один и тот
        же опыт даёт одни и те же векторы — библиотека обещает побайтовую
        воспроизводимость реплеям и тестам, и здесь это соблюдается.

        Проходом по эпизодам, а не по словарю: восприятие выводится из
        ОКРУЖЕНИЯ слова во фразе, а словарь окружения не хранит.
        """
        if self.perception is None:
            return
        try:
            rows = self.gate.episodic.searchable()
        except Exception:  # noqa: BLE001 — база может быть ещё пуста
            return
        for row in rows:
            tokens = self._tokenize_for_lexicon(row["context"])
            if tokens:
                self.perception.observe(tokens)
        if self.perception.exposures:
            logger.info(
                "[PERCEPTION] Восстановлено из памяти: %d слов, %d показов, "
                "зрелость %.2f",
                self.perception.vocabulary, self.perception.exposures,
                self.perception.maturity(self.settings.perception_maturity_target),
            )

    def get_or_create_brain_epoch(self, now: Optional[float] = None) -> float:
        """
        The origin of subjective time, one that SURVIVES a restart.

        Without it, every start-up set brain_time = time.time() and the
        clock jumped backwards relative to the last_decayed_at marks saved
        on the accelerated scale. Forgetting then switched off silently
        (_decay_nodes contains `if dt <= 0: continue`) — for another 2.3
        hours after a hundred-message conversation.

        Stored through the same meta-node mechanism as the sleep marker: a
        separate table for a single number would be excessive.
        """
        row = self.gate.semantic.meta("brain_epoch")
        if row is not None:
            try:
                return float(row["context"])
            except (TypeError, ValueError):
                logger.warning("[BRAIN EPOCH] Corrupted value — resetting")

        # `now` is supplied by the application from the same time source
        # the clock will use. Otherwise the epoch comes from a real
        # time.time() and differs between runs even when everything else
        # is pinned: the benchmark's replies diverged on the very first
        # message with a bit-identical graph and RNG state.
        epoch = now if now is not None else time.time()
        self.gate.semantic.upsert_meta(
            node_type="brain_epoch", content=str(epoch), weight=1.0, timestamp=epoch,
        )
        logger.info("[BRAIN EPOCH] Origin of subjective time: %.0f", epoch)
        return epoch


    def get_user_model_content(self) -> str:
        """Current text of the user model, falling back to the default."""
        row = self.gate.semantic.meta("user_model")
        return row["context"] if row is not None else self.settings.default_user_model



    # ----------------------------------------------------------------------
    # CONCEPT EXTRACTION — turning explanations into concepts
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    # LEXICAL ACQUISITION — learning a language from zero
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    # SURPRISE — the organism's own prediction error
    # ----------------------------------------------------------------------












    # ----------------------------------------------------------------------
    # 1. Storing a new link
    # ----------------------------------------------------------------------





    # ----------------------------------------------------------------------
    # 2. Finding similar context (keywords + fuzzy similarity)
    # ----------------------------------------------------------------------








    # ----------------------------------------------------------------------
    # 3. Updating last_accessed on a touch
    # ----------------------------------------------------------------------

    # Говорит ли человек О СЕБЕ. Русские и английские маркеры вместе:
    # ядро библиотеки языконезависимо, а витрина русская.
    _SELF_REFERENCE = re.compile(
        r"\b(я|мне|меня|мой|моя|моё|мои|мою|моего|моей|у меня|"
        r"i|me|my|mine|myself)\b",
        re.IGNORECASE,
    )




    # ----------------------------------------------------------------------
    # 3b. ASSOCIATIVE EDGES (semantic edges / spreading activation)
    # ----------------------------------------------------------------------




    # ----------------------------------------------------------------------
    # 4. Decay — the fading of old links
    # ----------------------------------------------------------------------




    # Lexical nodes are the infrastructure of language rather than
    # episodes of a conversation, so they live on their own, far longer
    # timescale.
    LEXICAL_NODE_TYPES = frozenset({"word", "syllable"})




    # ----------------------------------------------------------------------
    # 4b. SLEEP CYCLE — synaptic pruning and edge cleaning
    # ----------------------------------------------------------------------






    # ----------------------------------------------------------------------
    # 4c. SLEEP CYCLE — hub-and-spoke clustering
    # ----------------------------------------------------------------------



        # ----------------------------------------------------------------------
    # 4d. PROACTIVE MEMORY RECALL — choosing a node to speak up about
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    # 5. SELECTIVE CONSOLIDATION (short-term -> long-term)
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------





    def get_top_nodes(self, limit: int = 5) -> List[MemoryMatch]:
        rows = self.gate.node.all()
        nodes = [
            MemoryMatch(
                id=row["id"],
                context=row["context"],
                response=row["response"],
                weight=row["weight"],
                similarity=0.0,
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
            )
            for row in rows
        ]
        nodes.sort(key=lambda n: n.weight, reverse=True)
        return nodes[:limit]

    def count_nodes(self) -> int:
        return len(self.gate.node.all())

    def close(self) -> None:
        self.gate.close()

    def __enter__(self) -> "MemoryGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()