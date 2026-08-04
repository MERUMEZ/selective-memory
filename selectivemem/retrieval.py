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
 RETRIEVAL.PY — Как память отвечает
================================================================================
Поиск, предотбор кандидатов, оценка совпадения, переупорядочивание по
важности и растекание активации по связям.

ОЦЕНКА КАНДИДАТА, три составляющие плюс важность:

    релевантность = 0.3*общие_слова + 0.1*строковое_сходство + 0.5*семантика
    счёт          = min(1, релевантность + важность * 0.15)

Порог 0.3 разделяет ТОЛЬКО с работающим кодировщиком: с ним релевантное
даёт 0.30-0.65, постороннее 0.18-0.24. Без него 0.178 против 0.167 —
разделять нечего, и понижение порога вернёт шум вместо ответов. Поэтому
отсутствие семантики здесь один раз громко пишется в лог.

ВАЖНОСТЬ БЕРЁТСЯ ИЗ НАКОПЛЕННОЙ СИЛЫ, а не из веса: вес определяется
возрастом, и переупорядочивание по нему роняло R@1 с 32% до 18%.
_importance_base отвечает за верхнюю границу этого слагаемого, и там же
записано, чем кончились две прежние попытки нормировки.

ПРЕДОТБОР НЕ УКРАШЕНИЕ. Профилирование на 10 000 узлов: SequenceMatcher
съедал 82% времени поиска (4.49 с из 5.44), семантика — 6%. Дорогое
сравнение теперь идёт только по выжившим, и первый поиск ускорился с
781 мс до 14 мс.

ИЗВЛЕЧЕНИЕ ПОДКРЕПЛЯЕТ найденное, поэтому у search есть touch=False:
проверка на устаревание тоже зовёт поиск, и без флага КАЖДАЯ запись
незаметно подкрепляла соседей — замер поймал узлы силой 2.90, набранные
на служебных проверках.

Класс — миксин: состояние принадлежит MemoryGraph.
================================================================================
"""

from difflib import SequenceMatcher
import logging
from typing import Any, Dict, List, Optional, Set

from selectivemem import embeddings
# Выделение ключевых слов опирается на устройство языка, а не памяти:
# и шаблон слова, и список стоп-слов принадлежат словарю.
from selectivemem.neocortex import WORD_PATTERN
from selectivemem.stopwords import STOP_WORDS
from selectivemem.records import ActivationTrace, AssociatedNode, MemoryMatch

logger = logging.getLogger(__name__)


def _importance_base(own: float, headroom: float) -> float:
    """
    Накопленная сила -> слагаемое важности в оценке кандидата.

    ЖЁСТКИЙ ПОТОЛОК В ЕДИНИЦУ СТОЯЛ ЗДЕСЬ И ТИХО СЪЕДАЛ ДВЕ ТРЕТИ ШКАЛЫ.
    strength_max заявлен 3.0, но `min(1.0, own)` означал, что узел силой
    1.0 и узел силой 3.0 ранжируются одинаково. Пока сила росла медленно,
    это было незаметно: в потолок упирались только похвалённые узлы, и
    различение держалось на том, что остальные до него не дотягивали.

    Замерено на стенде порядка (24 соревнующихся узла, 2 сида):

        шаг 0.05   похвалённые 100% в потолке, обычные 25%   разрыв MRR +0.050
        шаг 0.15   похвалённые 100%,           обычные 67%   разрыв MRR +0.028

    То есть стоило подкреплению заработать, как в потолок упёрлись ОБЕ
    группы, и различать стало нечем. Дефект был всё это время; поднятый
    шаг лишь сделал его видимым.

    Здесь мягкое плечо: ниже единицы всё как было — это важно, иначе
    поедут пороги и пример из README, который уже дважды ломался о
    правки этой самой формулы. Выше единицы рост продолжается, но с
    коэффициентом headroom, чтобы сильный узел не забивал релевантность
    совсем.

    Оценка остаётся ЛОКАЛЬНОЙ: она зависит только от самого узла, а не от
    того, кто ещё попал в кандидаты. Нормировка по максимуму среди
    кандидатов это свойство ломала, и её поймал
    test_prefilter_agrees_on_top_three.
    """
    if own <= 1.0:
        return own
    return 1.0 + (own - 1.0) * headroom

class RetrievalMixin:
    """Поиск, оценка, переупорядочивание, растекание активации."""

    def _prefilter(self, rows, query_keywords, query_vector, top_k: int):
        """
        A cheap pre-filter over candidates: keywords, semantics, weight.

        Returns a list of (row, keyword_score, semantic_score) — the
        already-computed values travel onward so nothing is calculated
        twice.

        COSINES ARE COMPUTED AS ONE MATRIX. Calling numpy once per node
        spent more time on the calls than on the arithmetic; a single
        matrix multiplication does the same work at once. If numpy is
        unavailable or the nodes have no vectors, the semantic part simply
        drops to zero and the filter falls back to keywords and weight —
        the same gentle degradation as everywhere else in this module.
        """
        candidate_limit = max(
            top_k * self.settings.search_candidate_multiplier,
            self.settings.search_candidate_minimum,
        )

        keyword_scores = [
            self._keyword_overlap(query_keywords, self._extract_keywords(
                row["context"].strip().lower()
            ))
            for row in rows
        ]

        semantic_scores = [0.0] * len(rows)
        if query_vector is not None and rows:
            np = embeddings._numpy()
            if np is None:
                # Без numpy семантика раньше молча обнулялась, и поиск
                # деградировал до строкового БЕЗ ЕДИНОГО СЛОВА в логе — при
                # том что энкодер работал и вектора были. Считаем поштучно:
                # медленнее, но отвечает на вопрос, который задали.
                for index, row in enumerate(rows):
                    vector = self._node_vector(row)
                    if vector is not None:
                        semantic_scores[index] = max(
                            0.0, embeddings.cosine(query_vector, vector)
                        )
            else:
                vectors = [self._node_vector(row) for row in rows]
                known = [i for i, v in enumerate(vectors) if v is not None]
                if known:
                    matrix = np.asarray([vectors[i] for i in known], dtype=np.float32)
                    q = np.asarray(query_vector, dtype=np.float32)
                    norms = np.linalg.norm(matrix, axis=1) * float(np.linalg.norm(q))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sims = np.where(norms > 0.0, matrix @ q / norms, 0.0)
                    for position, index in enumerate(known):
                        semantic_scores[index] = max(0.0, float(sims[position]))

        if len(rows) <= candidate_limit:
            return list(zip(rows, keyword_scores, semantic_scores))

        # The pre-filter's ordering mirrors the final formula minus the
        # fuzzy component — otherwise candidates would be chosen by one
        # criterion and ranked by another.
        ranked = sorted(
            range(len(rows)),
            key=lambda i: (
                keyword_scores[i] * self.settings.memory_keyword_weight
                + semantic_scores[i] * self.settings.memory_semantic_weight
                + rows[i]["weight"] * self.settings.memory_weight_influence
            ),
            reverse=True,
        )[:candidate_limit]
        return [(rows[i], keyword_scores[i], semantic_scores[i]) for i in ranked]

    def search(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 1,
        timestamp: Optional[float] = None,
        with_associations: bool = True,
        touch: bool = True,
    ) -> List[MemoryMatch]:
        """
        The main search over the memory graph (keyword overlap + fuzzy
        similarity + semantics).

        With with_associations=True, spreading activation runs after the
        top_k nodes are chosen: for each match, adjacent nodes are pulled
        in through edges with weight >= edge_activation_threshold. Those
        associative nodes join the result as MemoryMatch entries whose
        similarity is the activation score, weaker than their source.
        """
        effective_threshold = threshold if threshold is not None else self.settings.memory_search_threshold

        query_normalized = query.strip().lower()
        if not query_normalized:
            return []

        query_keywords = self._extract_keywords(query_normalized)
        rows = self.db.fetch_searchable_nodes()

        scored: List[MemoryMatch] = []

        # The query vector is computed ONCE per search. None means
        # semantics is unavailable (no model, no library), and only the
        # string components remain, as before.
        query_vector = self._encode(query)

        # WARN ONCE PER GRAPH LIFETIME when semantics is missing.
        #
        # The search threshold (memory_search_threshold) separates
        # correctly ONLY with a working encoder. Measured in English: with
        # one, relevant queries score 0.30-0.65 and irrelevant 0.18-0.24,
        # and the 0.3 threshold sits exactly between. WITHOUT one it is
        # 0.178 against 0.167 — nothing to separate, and lowering the
        # threshold returns noise rather than answers.
        #
        # Staying silent about it is not an option: the caller gets
        # emptiness with no hint as to why. That is exactly how half a day
        # went on the benchmark before it became clear the search was not
        # broken, merely blind.
        if query_vector is None and not self._warned_no_semantics:
            self._warned_no_semantics = True
            logger.warning(
                "[SEARCH] No semantics available — matching by shared words only. "
                "A query worded differently from the stored text will find nothing. "
                "Attach an encoder: MemoryGraph(encoder=...) or "
                "pip install selective-memory[semantic] ([semantic-ru] for Russian)"
            )

        # ------------------------------------------------------------------
        # CANDIDATE PRE-FILTER. The expensive comparison runs only on survivors.
        #
        # Profiled over 10,000 nodes: SequenceMatcher ate 82% of search
        # time (4.49 s out of 5.44) while semantics took 6%. The most
        # expensive component was precisely the one that once let "skin"
        # beat "cat" — character-level similarity.
        #
        # So the cheap signals — keyword overlap, cosine, weight — are
        # computed over ALL nodes, and fuzzy similarity only over the best
        # candidates. That matches what it does anyway: it does not find
        # anything new, it refines the order among the already plausible.
        #
        # The candidate pool is deliberately generous (twenty times what
        # was asked for, but never fewer than fifty): a narrow filter would
        # save pennies while risking the loss of a node that fuzzy
        # similarity alone would have rescued.
        # ------------------------------------------------------------------
        prefiltered = self._prefilter(rows, query_keywords, query_vector, top_k)

        # РЕДКОЕ СЛОВО РАЗЛИЧАЕТ, ЧАСТОЕ — НЕТ, и без этого поправка на
        # общий оборот речи невозможна. Замер на LongMemEval показал, чем
        # проигрывает верная запись:
        #
        #   вопрос  "I'm planning a trip to DENVER..."
        #   первым  "I'm planning a trip with friends"
        #   вторым  "I'm planning a trip to California"
        #   улика   "During my previous visit to Denver..."  — третья
        #
        # Побеждает общий зачин, проигрывает единственное слово по делу.
        # Совпадение по словам считало их поровну.
        #
        # Частота берётся ПО КАНДИДАТАМ, а не по всей базе: это те, кто уже
        # откликнулся на запрос, и различать надо именно их. Слово, общее
        # для всей выдачи, не говорит ни о чём.
        idf = self._keyword_idf(rows, query_keywords)

        for row, keyword_score, semantic_score in prefiltered:
            if idf is not None:
                keyword_score = self._keyword_overlap(
                    query_keywords,
                    self._extract_keywords(row["context"].strip().lower()),
                    idf=idf,
                )
            context_normalized = row["context"].strip().lower()
            fuzzy_score = self._compute_fuzzy_similarity(query_normalized, context_normalized)

            # RELEVANCE — how well this answers the QUESTION. Nothing about
            # how dear the memory is: that is decided afterwards, among the
            # candidates that already fit (see _rerank_by_importance).
            relevance = (
                keyword_score * self.settings.memory_keyword_weight
                + fuzzy_score * self.settings.memory_fuzzy_weight
                + semantic_score * self.settings.memory_semantic_weight
            )
            if self.settings.rerank_band > 0.0:
                combined_score = min(1.0, relevance)
            else:
                # Old behaviour: importance is a summand competing with
                # relevance. Measured on tools/compare_ordering.py, raising
                # its share made BOTH groups worse — a heavy node floats to
                # the top of every query, including the ones it answers
                # wrongly. Kept as the default until the re-rank is proven.
                # В модели интерференции слагаемым идёт НАКОПЛЕННАЯ СИЛА, а
                # не вес. Разница принципиальная: вес определяется возрастом
                # (затухание встроено), поэтому он тянул выдачу к свежему —
                # замерено, переупорядочивание по нему роняло R@1 с 32% до
                # 18%. Сила часам не подчиняется и меняется только от
                # подкрепления, пользы и ПОДАВЛЕНИЯ при проигрыше.
                #
                # Без этого провода подавление конкурентов бессмысленно:
                # сила менялась бы, ни на что не влияя.
                if self.settings.use_relative_strength:
                    # ОГРАНИЧЕНИЕ, А НЕ НОРМИРОВКА, и обе прежние попытки
                    # были хуже.
                    #
                    # Деление на strength_max усыхало слагаемое вдвое:
                    # свежий узел силой 1.45 давал 0.48 вместо прежних
                    # 0.95, и пример из README переставал находиться на
                    # чистой установке — счёт 0.23 при пороге 0.3. Поймала
                    # это только проверка колеса в пустом окружении.
                    #
                    # Деление на максимум СРЕДИ КАНДИДАТОВ чинило пример,
                    # но делало оценку нелокальной: убрали кандидата —
                    # изменились счета у всех, и предотбор переставал
                    # сходиться с полным перебором. Это поймал
                    # test_prefilter_agrees_on_top_three.
                    #
                    # Ограничение сверху локально и сохраняет прежний
                    # масштаб: свежий узел даёт свои 0.95, подкреплённый
                    # упирается в единицу.
                    own = (row["strength"] if row["strength"] is not None
                           else row["weight"])
                    base = _importance_base(own, self.settings.strength_headroom)
                else:
                    base = row["weight"]
                if self.settings.importance_scales_relevance:
                    # МОДУЛЯЦИЯ ВМЕСТО СЛОЖЕНИЯ, и это третий раз, когда в
                    # этом проекте побеждает та же форма.
                    #
                    # Сложение упирается в min(1.0, ...): чем весомее
                    # важность, тем чаще сумма переваливает за единицу, и
                    # кандидаты схлопываются в один балл. Замерено прямо:
                    # доля важности 0.15 -> 73.3%, 0.40 -> 30.0%,
                    # 0.80 -> 13.3%. Канал нельзя расширить не потому, что
                    # важность вредна, а потому что расти ей некуда.
                    #
                    # Умножение сохраняет порядок по релевантности и лишь
                    # растягивает его: сильный узел обгоняет равного себе по
                    # смыслу, но не обгоняет того, кто отвечает лучше.
                    # Потолка нет, насыщения нет.
                    combined_score = relevance * (
                        1.0 + base * self.settings.memory_weight_influence
                    )
                else:
                    combined_score = min(
                        1.0,
                        relevance + base * self.settings.memory_weight_influence,
                    )

            if combined_score >= effective_threshold:
                scored.append(
                    MemoryMatch(
                        id=row["id"],
                        context=row["context"],
                        response=row["response"],
                        weight=row["weight"],
                        similarity=combined_score,
                        created_at=row["created_at"],
                        last_accessed=row["last_accessed"],
                    )
                )

        scored.sort(key=lambda m: m.similarity, reverse=True)
        scored = self._rerank_by_importance(scored)
        top_matches = scored[:top_k]

        # ВНУТРЕННЯЯ ПРОВЕРКА — НЕ ИСПОЛЬЗОВАНИЕ. Поиск растит силу
        # найденному: извлечение и есть доказательство пользы. Но
        # вытеснение устаревшего тоже зовёт поиск, и без этого флага
        # КАЖДАЯ ЗАПИСЬ незаметно подкрепляла соседей.
        #
        # Замер поймал это тестом: похвалённый узел вытеснялся толпой
        # проходных, набравших силу 2.90 против его собственной, — их
        # трогали десятки раз при проверках на противоречие.
        if touch:
            for match in top_matches:
                self.touch_node(match.id, timestamp=timestamp)

        # ПОДАВЛЕНИЕ КОНКУРЕНТОВ. Кандидаты, которые прошли порог, но в
        # выдачу не попали, слабеют.
        #
        # В психологии памяти это вызванное забывание: извлечение одного
        # следа активно ТОРМОЗИТ соседние по признаку, и именно поэтому
        # нужное остаётся находимым среди похожих. Забывание существует не
        # ради места — долговременная память не переполняется, — а чтобы
        # извлечение оставалось возможным.
        #
        # Замер, ради которого это сделано: на 50 почти-двойниках R@1
        # падает со 100% до 50%, на 800 держится 50% при R@5 83.3%.
        # Нужный узел ЛЕЖИТ в выдаче, но не первым — его топят соседи с
        # теми же словами.
        #
        # Шаг маленький намеренно: одно извлечение не должно ничего
        # решать. И риск честный — подавление ЗАКРЕПЛЯЕТ нынешний порядок,
        # так что ошибка первого ранга самоподдерживается. У людей ровно
        # так же; лечится это подкреплением, которое сильнее шага.
        if self.settings.retrieval_suppression > 0.0 and top_matches:
            winners = {m.id for m in top_matches}
            losers = [m.id for m in scored if m.id not in winners]
            for node_id in losers[: self.settings.retrieval_suppression_limit]:
                self.db.add_strength(
                    node_id,
                    -self.settings.retrieval_suppression,
                    self.settings.strength_max,
                )
            if losers:
                logger.debug(
                    "[SUPPRESSION] %d competitors weakened after retrieval",
                    min(len(losers), self.settings.retrieval_suppression_limit),
                )

        if top_matches:
            logger.info(
                "[MEMORY HIT] %d matches for %r (best score=%.3f, id=%s)",
                len(top_matches), query[:50], top_matches[0].similarity, top_matches[0].id,
            )
        else:
            logger.info("[MEMORY MISS] No matches for %r", query[:50])
            return top_matches

        # ------------------------------------------------------------------
        # SPREADING ACTIVATION (Multi-hop RAG)
        # ------------------------------------------------------------------
        self.last_activation_traces = []

        if with_associations:
            existing_ids = {m.id for m in top_matches}
            associative_extras: List[MemoryMatch] = []

            for source_match in top_matches:
                associated = self.get_associated_nodes(
                    source_match.id,
                    min_weight=self.settings.edge_activation_threshold,
                    limit=self.settings.edge_max_hop_nodes,
                    timestamp=timestamp,
                )

                for assoc in associated:
                    if assoc.id in existing_ids:
                        continue

                    activation_score = min(
                        1.0,
                        source_match.similarity * self.settings.edge_activation_decay * assoc.edge_weight,
                    )

                    logger.info(
                        "[ASSOCIATION] Node %s -> Node %s (edge_weight=%.2f, activation_score=%.3f)",
                        source_match.id, assoc.id, assoc.edge_weight, activation_score,
                    )

                    self.last_activation_traces.append(
                        ActivationTrace(
                            source_id=source_match.id,
                            target_id=assoc.id,
                            edge_weight=assoc.edge_weight,
                            activation_score=activation_score,
                        )
                    )

                    associative_extras.append(
                        MemoryMatch(
                            id=assoc.id,
                            context=assoc.context,
                            response=assoc.response,
                            weight=assoc.weight,
                            similarity=activation_score,
                            created_at=0.0,
                            last_accessed=0.0,
                        )
                    )
                    existing_ids.add(assoc.id)

            if associative_extras:
                top_matches = top_matches + associative_extras

        return top_matches

    def _node_vector(self, row):
        """
        A node's meaning vector, computed LAZILY.

        Nodes created before the model existed arrive with embedding=NULL.
        Rather than one heavy migration of the whole database, the vector
        is computed the first time the node is touched and stored right
        away; afterwards it is simply read.

        A node's meaning is taken from BOTH halves: the user may have
        asked in one set of words while the substance ended up in the
        bot's reply.
        """
        vector = embeddings.from_blob(row["embedding"])
        # THE DIMENSION IS CHECKED, and that is not pedantry. The encoder
        # is pluggable, so a database filled with one model's vectors will
        # one day be opened with another. A BLOB is read as a float32
        # array with no metadata whatsoever, so a foreign vector raises no
        # error — it quietly yields a meaningless similarity. Such a
        # mismatch shows up not as a crash but as search returning the
        # wrong things, and only measurement catches it.
        if vector is not None and (
            self._vector_dim is None or len(vector) == self._vector_dim
        ):
            return vector

        text = f"{row['context'] or ''} {row['response'] or ''}".strip()
        vector = self._encode(text)
        if vector is None:
            return None

        self.db.update_embedding(row["id"], embeddings.to_blob(vector))
        return vector

    def _encode(self, text: str):
        """The meaning vector through the attached encoder, or None."""
        encode = self.encoder if self.encoder is not None else embeddings.encode
        vector = encode(text)
        if vector is not None and self._vector_dim is None:
            self._vector_dim = len(vector)
        return vector

    def _extract_keywords(self, text: str) -> Set[str]:
        words = WORD_PATTERN.findall(text)
        return {
            w for w in words
            if len(w) >= self.settings.memory_min_keyword_length and w not in STOP_WORDS
        }

    @staticmethod
    def _keyword_overlap(query_keywords: Set[str], context_keywords: Set[str],
                         idf: Optional[Dict[str, float]] = None) -> float:
        """
        Доля совпавших слов. С idf — доля совпавшей РАЗЛИЧАЮЩЕЙ СИЛЫ.

        Без взвешивания «Denver» и «planning» стоят одинаково, и запись,
        разделяющая с вопросом только общий зачин, обходит ту, что
        разделяет единственное слово по делу.
        """
        if not query_keywords or not context_keywords:
            return 0.0
        intersection = query_keywords & context_keywords
        if idf is None:
            smallest_set_size = min(len(query_keywords), len(context_keywords))
            return len(intersection) / smallest_set_size if smallest_set_size else 0.0
        total = sum(idf.get(w, 1.0) for w in query_keywords)
        if total <= 0.0:
            return 0.0
        return sum(idf.get(w, 1.0) for w in intersection) / total

    def _keyword_idf(self, rows, query_keywords: Set[str]):
        """
        Различающая сила каждого слова запроса ПО ВСЕЙ ВЫДАЧЕ.

        idf(слово) = log(1 + N / сколько записей его содержат)

        Слово, встречающееся у всех, получает около нуля и перестаёт
        решать; слово у одного-двух — максимум.

        ПОЧЕМУ ПО ВСЕЙ БАЗЕ, А НЕ ПО КАНДИДАТАМ ПРЕДОТБОРА. Первая версия
        считала частоту по кандидатам, и это делало оценку НЕЛОКАЛЬНОЙ:
        счёт узла зависел от того, кто ещё попал в предотбор, поэтому
        предотбор переставал сходиться с полным перебором. Поймал это
        test_prefilter_agrees_on_top_three — тот же тест, что раньше уже
        ловил нормировку силы по максимуму среди кандидатов. Один и тот же
        соблазн: посчитать что-нибудь «среди тех, кто дошёл».

        ПРОШЛАЯ ПОПЫТКА БЫЛА ОТВЕРГНУТА И ОТВЕРГНУТА ЗРЯ. Её мерили на
        probe_semantic — шестнадцать фактов, — где частота слова
        бессмысленна по построению: почти всякое слово встречается один
        раз. Вывод «не изменило ни одного попадания» был верен для того
        стенда и неприменим к стогам в сотни реплик.
        """
        if not self.settings.keyword_idf or not query_keywords or not rows:
            return None
        import math

        # ПО ИНДЕКСУ, А НЕ ПЕРЕБОРОМ. Перебор всех записей ради частоты
        # съедал выигрыш предотбора: 753 мс против 2206 мс на трёх тысячах
        # узлов. Полнотекстовый индекс отвечает одним COUNT на слово.
        counts = self.db.document_frequency(sorted(query_keywords))
        total = len(rows)
        return {w: math.log(1.0 + total / max(1, c)) for w, c in counts.items()}

    @staticmethod
    def _compute_fuzzy_similarity(query: str, context: str) -> float:
        return SequenceMatcher(None, query, context).ratio()

    def _importance_scores(self, matches: List["MemoryMatch"]) -> Dict[int, float]:
        """
        How DEAR each memory is, independently of any question.

        Signals are summed with configurable weights; all but the node's
        weight are off by default. They are switched on ONE AT A TIME and
        measured, because this project has already been taught the
        difference between a signal that sounds right and one that works:
        spike strength looked like the perfect measure of importance and
        turned out to measure novelty.

        Everything needed beyond MemoryMatch is fetched in batch: this runs
        inside search, on every recall.
        """
        settings = self.settings
        ids = [m.id for m in matches]

        degrees: Dict[int, int] = {}
        if settings.importance_connectivity > 0.0:
            degrees = self.db.get_degrees(ids)

        rows: Dict[int, Any] = {}
        if settings.importance_use > 0.0 or settings.use_relative_strength:
            rows = {row["id"]: row for row in self.db.get_nodes_by_ids(ids)}

        # ДОЛЯ вместо веса. Считается по кандидатам, а не по всей базе:
        # нормировать при записи значило бы трогать каждый узел на каждое
        # сообщение, а сравнивать надо ровно тех, кто конкурирует за одну
        # выдачу.
        shares: Dict[int, float] = {}
        if settings.use_relative_strength:
            raw = {
                match.id: max(0.0, (rows[match.id]["strength"] if match.id in rows else None)
                              or match.weight)
                for match in matches
            }
            total = sum(raw.values())
            if total > 0:
                shares = {node_id: value / total for node_id, value in raw.items()}

        scores: Dict[int, float] = {}
        for match in matches:
            # Вес узла определяется ВОЗРАСТОМ (затухание встроено в него), и
            # замер показал цену: переупорядочивание по нему на внешнем
            # наборе роняло R@1 с 32% до 18%, потому что сортировка "по
            # важности" оказывалась сортировкой по свежести. Доля от
            # накопленной силы возрасту не подчиняется.
            base = shares.get(match.id, match.weight) if shares else match.weight
            score = base * settings.importance_weight_signal

            if settings.importance_connectivity > 0.0:
                degree = degrees.get(match.id, 0)
                normalised = min(1.0, degree / max(1, settings.importance_degree_full))
                score += normalised * settings.importance_connectivity

            if settings.importance_self_reference > 0.0:
                if self._SELF_REFERENCE.search(match.context or ""):
                    score += settings.importance_self_reference

            if settings.importance_use > 0.0:
                row = rows.get(match.id)
                stability = (row["stability"] if row else None) or settings.stability_initial
                normalised = min(1.0, stability / max(1e-9, settings.stability_max))
                score += normalised * settings.importance_use

            scores[match.id] = score
        return scores

    def _rerank_by_importance(self, scored: List["MemoryMatch"]) -> List["MemoryMatch"]:
        """
        Reorders the HEAD of a relevance-sorted list by importance.

        Why a separate stage rather than one more summand. Importance as a
        summand competes with relevance, and measurement showed what that
        costs: raising the weight's share from 0.15 to 0.70 dropped the
        praised group's MRR from 0.950 to 0.649 and the ordinary group's
        from 0.877 to 0.751. A heavy node rises to the top of EVERY query,
        including those it answers wrongly — it displaces correct ordinary
        answers and buries its own correct ones under other heavy nodes.
        Both groups lose, and the gap collapses.

        So importance never promotes anything irrelevant. It only decides
        the order among answers that are ALREADY about equally fitting —
        those within rerank_band of the best relevance. Everything below the
        band keeps its relevance order.

        rerank_band = 0 disables the stage entirely and restores the old
        summand behaviour.
        """
        band = self.settings.rerank_band
        if band <= 0.0 or len(scored) < 2:
            return scored

        cutoff = scored[0].similarity - band
        head = [m for m in scored if m.similarity >= cutoff]
        tail = [m for m in scored if m.similarity < cutoff]

        importance = self._importance_scores(head)
        head.sort(key=lambda m: importance[m.id], reverse=True)

        logger.debug(
            "[RERANK] %d of %d candidates within band %.2f reordered by importance",
            len(head), len(scored), band,
        )
        return head + tail

    def get_associated_nodes(
        self,
        node_id: int,
        min_weight: Optional[float] = None,
        limit: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> List[AssociatedNode]:
        """
        Nodes adjacent to node_id through edges with weight >= min_weight,
        sorted by edge weight — the strongest associations first.

        Used by search() for spreading activation.
        """
        effective_min_weight = min_weight if min_weight is not None else self.settings.edge_activation_threshold

        edge_rows = self.db.get_edges_for_node(node_id)
        strong_edges = [row for row in edge_rows if row["weight"] >= effective_min_weight]
        strong_edges.sort(key=lambda r: r["weight"], reverse=True)

        if limit is not None:
            strong_edges = strong_edges[:limit]

        results: List[AssociatedNode] = []
        for edge_row in strong_edges:
            neighbor_row = self.db.get_node(edge_row["neighbor_id"])
            if neighbor_row is None:
                continue

            results.append(
                AssociatedNode(
                    id=neighbor_row["id"],
                    context=neighbor_row["context"],
                    response=neighbor_row["response"],
                    weight=neighbor_row["weight"],
                    edge_weight=edge_row["weight"],
                    activation_score=edge_row["weight"],
                    source_node_id=node_id,
                )
            )
            # Reaching an associative node counts as a touch too: it was
            # recalled, even though it was not found directly by score.
            self.touch_node(neighbor_row["id"], timestamp=timestamp)

            # AND THE ROAD TAKEN IS KEPT CLEAR. Spreading activation used to
            # read an edge's weight and never write to it, so an edge had no
            # way of registering that it had been USEFUL. Only co-occurrence
            # at write time could strengthen one.
            #
            # That is what made aggressive pruning impossible: cutting edges
            # hard would have severed the working paths along with the dead
            # ones, because nothing told them apart.
            #
            # Biology puts the limit here rather than on storage. Long-term
            # memory does not fill up; what the brain prunes, expensively and
            # continuously, is CONNECTIONS. A memory is lost by becoming
            # unreachable, not by being erased — the classmate's name you
            # cannot recall but recognise on sight.
            if self.settings.edge_use_boost > 0.0:
                self.connect_nodes(
                    node_id, neighbor_row["id"],
                    weight_boost=self.settings.edge_use_boost,
                    timestamp=timestamp,
                    edge_type="association",
                )

        return results
