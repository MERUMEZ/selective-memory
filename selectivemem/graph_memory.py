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
 GRAPH_MEMORY.PY — Гиппокамп: запись эпизода и связи между эпизодами
================================================================================
Здесь осталось то, что в мозге делает гиппокамп: быстрая запись нового
эпизода, обнаружение устаревшего, связывание одновременно активного и
подкрепление по награде. Остальное разнесено по участкам:

    lexicon.py        словарь и собственное удивление (кора)
    retrieval.py      поиск, оценка, растекание активации
    consolidation.py  сон: реактивация, понижение, подрезка, абстракция
    forgetting.py     затухание узлов и связей, ёмкость
    records.py        структуры ответа
    stopwords.py      служебные слова

ЗАЧЕМ РАЗНОСИЛИ. Файл был на 2666 строк и держал всё сразу, и это не
стилистика: пока стадии сна лежали вперемешку с записью и поиском, никому
не бросалось в глаза, что двух первых стадий у сна нет вовсе. А словарь
жил в одной таблице с воспоминаниями, из-за чего подсчёт узлов включал
лексику, порог пробивался на девятом сообщении и сон запускался на каждое
следующее — живой дефект, стоивший двух вызовов языковой модели на
реплику.

РАЗДЕЛЕНИЕ ПОКА ФАЙЛОВОЕ. Модули — миксины: состояние (db, settings,
encoder) принадлежит MemoryGraph, они его подмешивают. Публичный интерфейс
не изменился ни на имя, и это было условием: снаружи граф зовут тесты,
стенды и приложения. Переход на владение данными по участкам (composition)
ломает вызовы и делается отдельно, с полным перемером.

КОНСОЛИДАЦИЯ ЭПИЗОДА (consolidate_from_stm, теперь в consolidation.py)
судит накопленный буфер по двум признакам и приходит к одному из трёх
решений: эмоциональный узел, структурный узел или рутинный шум, который не
записывается вовсе.
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
# Забывание — в forgetting.py. Там же записано, что теорий в нём сейчас
# две: ранжирование живёт по интерференции, а само затухание всё ещё
# распад по часам. Переход остановлен на середине сознательно.
from selectivemem.forgetting import ForgettingMixin
# Извлечение — в retrieval.py: поиск, предотбор, оценка, растекание.
from selectivemem.retrieval import RetrievalMixin
from selectivemem.lexicon import VOWELS, WORD_PATTERN, LexiconMixin  # noqa: F401
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
    from selectivemem.working_memory import STMEntry

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




class MemoryGraph(ConsolidationMixin, ForgettingMixin, LexiconMixin, RetrievalMixin):
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
        row = self.db.get_meta_node("brain_epoch")
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
        self.db.upsert_meta_node(
            node_type="brain_epoch", content=str(epoch), weight=1.0, timestamp=epoch,
        )
        logger.info("[BRAIN EPOCH] Origin of subjective time: %.0f", epoch)
        return epoch


    def get_user_model_content(self) -> str:
        """Current text of the user model, falling back to the default."""
        row = self.db.get_meta_node("user_model")
        return row["context"] if row is not None else self.settings.default_user_model



    # ----------------------------------------------------------------------
    # CONCEPT EXTRACTION — turning explanations into concepts
    # ----------------------------------------------------------------------

    def create_concept_node(
        self,
        name: str,
        definition: str,
        source_node_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Creates a concept node — or updates it if the concept is already
        known — and wires it into the knowledge graph:

            1. The node is stored through upsert_concept_node
               (node_type='concept').
            2. An edge concept <-> user node is created or strengthened:
               the user is the source of this knowledge.
            3. If semantically similar nodes already exist, initial
               associative edges concept <-> similar_node are laid down,
               up to concept_max_similar_links of them.
            4. If source_node_id is given — say the episode the concept was
               extracted from — the concept is linked to it as well.

        Returns the id of the concept node.
        """
        ts = timestamp if timestamp is not None else time.time()
        normalized_name = name.strip()

        concept_node_id, was_created = self.db.upsert_concept_node(
            name=normalized_name,
            definition=definition.strip(),
            weight=self.settings.concept_node_weight,
            timestamp=ts,
        )

        # --- Link to the user model: the source of this knowledge ---
        user_row = self.db.get_meta_node("user_model")
        if user_row is not None:
            self.connect_nodes(
                concept_node_id,
                user_row["id"],
                weight_boost=self.settings.concept_user_edge_weight,
                timestamp=ts,
            )

        # --- Link to the originating node, when one was given ---
        if source_node_id is not None:
            self.connect_nodes(
                concept_node_id,
                source_node_id,
                weight_boost=self.settings.edge_initial_weight,
                timestamp=ts,
            )

        # --- Semantic linking to similar existing nodes ---
        if was_created:
            self._link_concept_to_similar_nodes(concept_node_id, normalized_name, definition, ts)

        logger.info(
            "[CONCEPT EXTRACTED] Node '%s' (type=concept) stored and linked to the user model",
            normalized_name,
        )

        return concept_node_id

    def _link_concept_to_similar_nodes(
        self,
        concept_node_id: int,
        name: str,
        definition: str,
        timestamp: float,
    ) -> None:
        """
        Looks for nodes that echo the new concept — matching on its name
        and definition — and lays down initial associative edges, up to
        concept_max_similar_links of them, excluding the concept itself.
        """
        query_text = f"{name} {definition}"
        matches = self.search(
            query_text,
            threshold=self.settings.concept_similarity_link_threshold,
            top_k=self.settings.concept_max_similar_links + 1,  # +1 in case it matches itself
            timestamp=timestamp,
            with_associations=False,
        )

        linked_count = 0
        for match in matches:
            if match.id == concept_node_id:
                continue
            if linked_count >= self.settings.concept_max_similar_links:
                break

            self.connect_nodes(
                concept_node_id,
                match.id,
                weight_boost=self.settings.concept_similarity_edge_weight,
                timestamp=timestamp,
            )
            linked_count += 1

        if linked_count:
            logger.info(
                "[CONCEPT LINKED] concept_id=%s linked to %d similar nodes",
                concept_node_id, linked_count,
            )

    # ----------------------------------------------------------------------
    # LEXICAL ACQUISITION — learning a language from zero
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    # SURPRISE — the organism's own prediction error
    # ----------------------------------------------------------------------












    # ----------------------------------------------------------------------
    # 1. Storing a new link
    # ----------------------------------------------------------------------

    def find_superseded(
        self,
        text: str,
        exclude_id: Optional[int] = None,
        explicit_correction: bool = False,
    ) -> List["SupersededNode"]:
        """
        Which existing memories a new one SUPERSEDES.

        Without this, memory piled up mutually exclusive facts and
        returned an arbitrary one: "my dog is called Rex", later "my dog is
        called Buddy" — both nodes equal, and the stale one actually
        scoring BETTER (0.906 against 0.875), because the ranking is
        decided by string similarity rather than by time.

        Supersession requires two conditions at once:
          1. high SEMANTIC similarity: they are about the same thing;
          2. INCOMPLETE word overlap: this is a different version, not a
             repetition. A plain repetition must simply reinforce the node.

        explicit_correction means the user corrected something outright
        ("no", "that's wrong"). That is strong evidence, so the topic
        threshold is lowered: without such a marker we are cautious, with
        one we trust.

        The threshold is deliberately high. Erring towards "missed a
        contradiction" is cheaper than weakening an independent memory —
        though even that is no catastrophe, because nodes are weakened
        rather than deleted (see supersede_node).
        """
        query_vector = self._encode(text)
        if query_vector is None:
            # Without semantics there is no way to tell "a different
            # version" from "a different subject": string similarity is
            # equally high for "called Rex"/"called Buddy" and for
            # "called Rex"/"called Rex".
            return []

        threshold = self.settings.contradiction_topic_threshold
        if explicit_correction:
            threshold -= self.settings.contradiction_correction_relief

        new_words = self._extract_keywords(text.lower())

        if self.settings.contradiction_search_threshold > 0.0:
            return self._superseded_via_search(
                text, new_words, exclude_id, explicit_correction
            )

        found: List[SupersededNode] = []

        # КАНДИДАТЫ ИЗ ИНДЕКСА, а не перебор всей базы. Проверка на
        # устаревание идёт при каждой записи, и полный скан с косинусом на
        # каждый узел делал запись невыносимой задолго до того, как
        # замедлялся поиск: три тысячи узлов не записывались за две минуты.
        #
        # Отбор по общим словам ничего не теряет: вытеснение всё равно
        # требует пересечения не ниже contradiction_min_overlap, то есть
        # узел без единого общего слова был бы отвергнут дальше.
        candidates = self.db.fetch_candidates_by_text(
            sorted(new_words), self.settings.supersede_scan_limit,
        )
        for row in candidates:
            if row["id"] == exclude_id or row["is_meta"]:
                continue

            # Compare ONLY what the user said, without the bot's replies.
            # _node_vector builds its vector from the question-and-answer
            # pair, which is right for search but wrong here: a fact lives
            # in what the PERSON said, while the bot's "got it" or "okay"
            # is noise that shifts the vector and decides the comparison.
            similarity = embeddings.cosine(
                query_vector, self._encode(row["context"] or "")
            )
            if similarity < threshold:
                continue

            old_words = self._extract_keywords((row["context"] or "").lower())
            overlap = self._keyword_overlap(new_words, old_words)
            if overlap >= self.settings.contradiction_repeat_threshold:
                continue  # a repetition, not a new version
            if overlap < self.settings.contradiction_min_overlap:
                # Слишком мало общих слов — защита от чужого кодировщика,
                # см. тот же охранник в _superseded_via_search. Здесь он
                # нужнее: этот путь стоит по умолчанию, и на английском
                # тексте с русской моделью он срабатывал 3080 раз на 79
                # записей, ослабляя по сорок узлов на каждую запись.
                continue

            found.append(
                SupersededNode(
                    id=row["id"],
                    context=row["context"],
                    similarity=similarity,
                    word_overlap=overlap,
                )
            )

        return found

    def _superseded_via_search(
        self,
        text: str,
        new_words: Set[str],
        exclude_id: Optional[int],
        explicit_correction: bool,
    ) -> List["SupersededNode"]:
        """
        Finds stale versions among what ORDINARY SEARCH returns, instead of
        scanning every node with a high cosine bar.

        WHY THE OLD WAY WAS BROKEN BY DESIGN, not by a badly chosen number.
        It compared whole sentences and demanded cosine >= 0.8. But a
        contradiction is "same subject, DIFFERENT value" — so the stronger
        the change, the less similar the sentences, and the more surely the
        update slips through. The evidence of a contradiction was being
        subtracted from the evidence of relatedness.

        Measured, against a threshold of 0.8:

            "мою собаку зовут Рекс" -> "... Бобик"        0.923  caught
            "моя собака зовут Рекс" -> "мою собаку ТЕПЕРЬ
                                        зовут Бобик"      0.722  missed
            "я живу в Москве" -> "я ПЕРЕЕХАЛ в Питер"     0.369  missed

        The first pair is the example from this method's own docstring. The
        mechanism was calibrated for restatements that swap a single word
        into the same template, and blind to how people actually report a
        change.

        Search finds the old node in ALL of those cases, because it blends
        keywords, fuzzy similarity and meaning rather than trusting one
        cosine. So the roles swap: SEARCH FINDS, and the word-overlap test
        decides whether this is a new version or a repetition.

        HONEST LIMIT. Five of six real updates separate cleanly — the worst
        scores 0.642 while the best unrelated memory scores 0.433. The
        sixth does not separate at all: "я живу в Москве" and "я переехал в
        Питер" share no words, and no measure of string similarity will
        connect them. That needs knowing that "moved" cancels "live in",
        which is knowledge this library does not have.
        """
        threshold = self.settings.contradiction_search_threshold
        if explicit_correction:
            threshold -= self.settings.contradiction_correction_relief

        candidates = self.search(
            text,
            top_k=self.settings.contradiction_candidates,
            with_associations=False,
            touch=False,          # проверка, а не использование
        )

        found: List[SupersededNode] = []
        for match in candidates:
            if match.id == exclude_id or match.similarity < threshold:
                continue

            old_words = self._extract_keywords((match.context or "").lower())
            overlap = self._keyword_overlap(new_words, old_words)
            if overlap >= self.settings.contradiction_repeat_threshold:
                continue          # повтор, а не новая версия
            if overlap < self.settings.contradiction_min_overlap:
                # СЛИШКОМ МАЛО ОБЩИХ СЛОВ — защита от чужого кодировщика.
                # Замер: русская модель (та, что в пакете) на английском
                # тексте даёт "my dog is called Rex" против "the price of
                # bread went up" косинус 0.808 при пороге 0.8. Порог сидит
                # внутри шума, и память начинает ослаблять факт про собаку,
                # потому что подорожал хлеб. На том же тексте прежний путь
                # срабатывал 3080 раз на 79 записей — сорок ослаблений на
                # каждую запись.
                #
                # Общее слово подделать эмбеддингом нельзя, поэтому проверка
                #языконезависима и стоит один проход по множеству.
                continue

            found.append(
                SupersededNode(
                    id=match.id,
                    context=match.context,
                    similarity=match.similarity,
                    word_overlap=overlap,
                )
            )
        return found

    def supersede_node(self, node_id: int, timestamp: Optional[float] = None) -> None:
        """
        Marks a memory as superseded: lowers its weight and RESETS its
        stability, returning the node to the forgettable pile.

        Weakening, deliberately, rather than deletion. If the fact was in
        truth still valid — a false positive such as "I have a cat" against
        "I have a dog" — the user will mention it again, the node will be
        touched and its stability will grow back. Deletion would be
        irreversible; here a mistake is cheap and repairs itself.
        """
        row = self.db.get_node(node_id)
        if row is None:
            return

        new_weight = max(0.0, row["weight"] - self.settings.contradiction_weight_penalty)
        stability = (row["stability"] or self.settings.stability_initial)
        new_stability = max(
            self.settings.stability_initial, stability * self.settings.contradiction_stability_factor
        )

        self.db.update_weight(node_id, new_weight)
        self.db.update_stability(node_id, new_stability)

        # ШТРАФ ОБЯЗАН ДОХОДИТЬ ДО ТОГО, ЧТО РЕШАЕТ ВЫДАЧУ. После перехода
        # на модель интерференции ранжирование смотрит на накопленную силу,
        # а вытеснение понижало только вес — и штраф перестал доходить.
        #
        # Тест поймал это сразу: на запрос "как зовут мою собаку" первым
        # снова шёл устаревший "Рекс" вместо актуального "Бобика", то есть
        # вернулась ровно та болезнь, ради которой механизм и написан.
        if self.settings.use_relative_strength:
            self.db.add_strength(
                node_id,
                -self.settings.contradiction_weight_penalty,
                self.settings.strength_max,
            )

        logger.info(
            "[SUPERSEDED] Node %s replaced by a newer version: weight %.3f -> %.3f, "
            "stability %.1f -> %.1f",
            node_id, row["weight"], new_weight, stability, new_stability,
        )

    def save_connection(
        self,
        context: str,
        response: str,
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
        explicit_correction: bool = False,
        node_type: str = "episodic",
    ) -> int:
        """
        Stores a new link context -> response with an initial weight.

        explicit_correction means the user corrected something outright
        ("no", "that's wrong"). It lowers the bar for superseding stale
        versions.
        """
        initial_weight = weight if weight is not None else self.settings.base_plasticity_threshold

        node_id = self.db.insert_node(
            context=context,
            response=response,
            weight=initial_weight,
            timestamp=timestamp,
            node_type=node_type,
        )

        # ВЕКТОР СЧИТАЕТСЯ ПРИ ЗАПИСИ, а не лениво при первом поиске.
        #
        # Ленивый расчёт растягивал стоимость на первое обращение, и
        # платил за него пользователь: замер показал 781 мс на первом
        # поиске по тысяче узлов против 14 мс на прогретом. То есть
        # человек, открывший бота утром, ждал почти секунду — а потом
        # всё летало, и в отчётах об ошибках это выглядело бы загадкой.
        #
        # При записи та же работа стоит один вызов кодировщика на узел и
        # размазана ровно там, где её ждут.
        if node_id is not None:
            vector = self._encode(f"{context} {response}".strip())
            if vector is not None:
                self.db.update_embedding(node_id, embeddings.to_blob(vector))

        # A newer version of a fact supersedes the older one: otherwise
        # memory piles up mutually exclusive nodes and returns an
        # arbitrary one.
        for stale in self.find_superseded(
            context, exclude_id=node_id, explicit_correction=explicit_correction
        ):
            logger.info(
                "[CONTRADICTION] %r supersedes %r (similarity %.2f, shared words %.2f)",
                context[:40], stale.context[:40], stale.similarity, stale.word_overlap,
            )
            self.supersede_node(stale.id, timestamp=timestamp)

        logger.info(
            "[SPIKE DETECTED] New link stored id=%s weight=%.3f",
            node_id, initial_weight,
        )
        return node_id

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



    def touch_node(self, node_id: int, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.db.update_last_accessed(node_id, timestamp=ts)
        logger.debug("[MEMORY TOUCHED] id=%s last_accessed updated (t=%.2f)", node_id, ts)

    # ----------------------------------------------------------------------
    # 3b. ASSOCIATIVE EDGES (semantic edges / spreading activation)
    # ----------------------------------------------------------------------

    def connect_nodes(
        self,
        node_from: int,
        node_to: int,
        weight_boost: Optional[float] = None,
        timestamp: Optional[float] = None,
        edge_type: Optional[str] = None,
    ) -> float:
        """
        Creates or strengthens an associative edge between two long-term
        nodes.

        Two scenarios use it:
            1. Contextual linking: node A was pulled from memory (a search
               hit) and node B was created or reinforced during the same
               exchange -> the edge A -> B is strengthened.
            2. Co-activation linking: several nodes were used within one
               STM window -> the edges between them grow (see
               reinforce_coactivation).

        The edge is ignored when node_from == node_to; a self-loop means
        nothing. Returns the resulting edge weight.
        """
        if node_from is None or node_to is None or node_from == node_to:
            return 0.0

        # Race protection: one of the nodes may have been deleted — a
        # low-weight syllable node caught by orphan pruning during sleep,
        # say — between the moment its id was recorded and this call. The
        # FOREIGN KEY on edges would otherwise blow up the insert, so the
        # edge is quietly skipped instead.
        if self.db.get_node(node_from) is None or self.db.get_node(node_to) is None:
            logger.debug(
                "[ASSOCIATION SKIP] Node %s or %s no longer exists (deleted) -> edge not created",
                node_from, node_to,
            )
            return 0.0

        boost = weight_boost if weight_boost is not None else self.settings.edge_boost_step
        ts = timestamp if timestamp is not None else time.time()

        new_weight = self.db.upsert_edge(
            node_from=node_from,
            node_to=node_to,
            weight_boost=boost,
            timestamp=ts,
            edge_type=edge_type,
        )

        logger.info(
            "[ASSOCIATION] Node %s -> Node %s (edge_weight=%.2f)",
            node_from, node_to, new_weight,
        )
        return new_weight

    def reinforce_coactivation(
        self,
        node_ids: List[int],
        weight_boost: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Co-activation linking: when several long-term nodes were touched
        or reinforced within ONE STM window, the edges between EVERY pair
        of them grow.

        node_ids is the list of nodes activated in the current window;
        duplicates and None values are filtered out automatically.
        """
        unique_ids = sorted({nid for nid in node_ids if nid is not None})
        if len(unique_ids) < 2:
            return

        boost = weight_boost if weight_boost is not None else self.settings.edge_boost_step
        ts = timestamp if timestamp is not None else time.time()

        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                self.connect_nodes(unique_ids[i], unique_ids[j], weight_boost=boost, timestamp=ts)

        logger.info(
            "[COACTIVATION] Edges reinforced between co-activated nodes: %s",
            unique_ids,
        )


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

    def reinforce_node(self, node_id: int, boost: float = 0.1, timestamp: Optional[float] = None) -> None:
        row = self.db.get_node(node_id)
        if row is None:
            logger.warning("[MEMORY REINFORCE] Node id=%s not found", node_id)
            return

        new_weight = min(1.0, row["weight"] + boost)
        self.db.update_weight(node_id, new_weight)
        self.touch_node(node_id, timestamp=timestamp)
        logger.info("[MEMORY REINFORCED] id=%s new weight=%.3f", node_id, new_weight)

    def apply_reward(
        self,
        node_id: int,
        valence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[RewardSignal]:
        """
        The dopamine signal: computes the REWARD PREDICTION ERROR for a
        node, updates its expectation and returns the result.

            rpe = actual valence - what this node expected
            expectation += reward_expectation_learning_rate * rpe

        This is the Rescorla-Wagner rule. The point of it: dopamine is
        released not by reward but by UNEXPECTED reward. Without it, the
        pursuit of approval degenerates — the organism would find one word
        that is always praised and repeat it forever. Here, what is
        ALWAYS praised stops producing a signal (rpe -> 0) and the
        organism goes off to try something new.

        Returns None if the node has disappeared: it may have been pruned
        between the action and the rating.
        """
        row = self.db.get_node(node_id)
        if row is None:
            return None

        expected = row["reward_expectation"] or 0.0
        rpe = valence - expected
        new_expectation = max(-1.0, min(1.0, expected + self.settings.reward_expectation_learning_rate * rpe))

        self.db.update_reward_expectation(node_id, new_expectation)
        # Одобрение поднимает НАКОПЛЕННУЮ СИЛУ, а не только ожидание
        # награды. Вес для этого не годится: он затухает от времени, и
        # через две недели от похвалы не остаётся следа — замерено,
        # похвалённый узел терял 0.95 -> 0.17 за месяц. Сила часам не
        # подчиняется, поэтому одобрение сохраняется столько, сколько
        # его не разбавили новые записи.
        self.db.add_strength(
            node_id,
            valence * self.settings.strength_reward_step,
            self.settings.strength_max,
        )

        logger.info(
            "[DOPAMINE] node=%s valence=%+.2f expected=%+.2f -> rpe=%+.2f "
            "(new expectation %+.2f)",
            node_id, valence, expected, rpe, new_expectation,
        )
        return RewardSignal(
            node_id=node_id,
            valence=valence,
            expected=expected,
            prediction_error=rpe,
            new_expectation=new_expectation,
        )

    def learning_scale(self, prediction_error: float) -> float:
        """
        By how much the reward prediction error accelerates consolidation.

        Dopamine modulates synaptic plasticity: an unexpected outcome
        consolidates strongly, a fully predicted one almost not at all.
        The lower bound (reward_min_learning_scale) keeps learning from
        reaching exactly zero — otherwise a long-mastered node would stop
        receiving even maintenance reinforcement.
        """
        return max(self.settings.reward_min_learning_scale, min(1.0, abs(prediction_error)))

    def penalize_node(
        self,
        node_id: int,
        penalty: float = 0.15,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Penalises a node for negative feedback: lowers its weight and
        deliberately does NOT move last_accessed forward, unlike
        touch_node or reinforce_node. That accelerates the node's relative
        ageing at the next apply_decay, modelling the lower durability of
        a negatively reinforced link.
        """
        row = self.db.get_node(node_id)
        if row is None:
            logger.warning("[MEMORY PENALIZE] Node id=%s not found", node_id)
            return

        new_weight = max(0.0, row["weight"] - penalty)
        self.db.update_weight(node_id, new_weight)
        logger.info(
            "[MEMORY PENALIZED] id=%s weight %.3f -> %.3f (penalty=%.3f)",
            node_id, row["weight"], new_weight, penalty,
        )

    def get_top_nodes(self, limit: int = 5) -> List[MemoryMatch]:
        rows = self.db.fetch_all_nodes()
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
        return len(self.db.fetch_all_nodes())

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MemoryGraph":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()