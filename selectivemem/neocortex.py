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
 NEOCORTEX.PY — Медленное знание: слова, понятия, схемы
================================================================================
Кора — не «второе хранилище воспоминаний», а вещество другого рода. Она
копит СТАТИСТИКУ: что с чем обычно встречается, что на что похоже, что
случается часто. Медленно, с перекрытием, без привязки к отдельному
случаю. Гиппокамп помнит «в четверг Лиза сказала про арахис», кора знает
«арахис — про аллергию».

ПОЭТОМУ ЗДЕСЬ ТРИ ВЕЩИ, КОТОРЫЕ РАНЬШЕ ЛЕЖАЛИ В ТРЁХ ФАЙЛАХ:

    словарь и слоги      — статистика языка, накопленная из всего виденного;
    собственное удивление — оценка новизны ПО ЭТОЙ статистике;
    понятия и схемы      — обобщения, в том числе свёртки плотных кластеров.

Разносить их было ошибкой разделения по функциям вместо разделения по
веществу: словарь оказывался «лексикой», схема — «сном», понятие —
«графом», хотя это одно и то же медленно накапливаемое знание.

СХЕМА ПРИНАДЛЕЖИТ КОРЕ, А НЕ СНУ. Сон её запускает — реактивация строит
структуру, из которой схема вырастает, — но живёт и работает она здесь.
create_abstract_node зовётся из consolidation.py именно поэтому: там
только повод, здесь владелец.

УДИВЛЕНИЕ СЧИТАЕТСЯ ЗДЕСЬ, И ЭТО БИОЛОГИЧЕСКИ СПОРНО. В мозге
рассогласование ожидаемого с пришедшим замечает гиппокамп (поле CA1), а
кора поставляет ему ожидания. У нас новизна целиком выводится из корковой
статистики слов, и это ровно та причина, по которой перефразировка
знакомой мысли удивляет сильнее, чем должна: сравниваются буквы, а не
положения дел. Известный предел, записанный честно.

Класс — миксин: состояние принадлежит MemoryGraph.
================================================================================
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from selectivemem.records import (
    HubCluster,
    KnownSyllable,
    KnownWord,
    LexicalProcessingResult,
    SurpriseResult,
)

logger = logging.getLogger(__name__)

# Слово — последовательность буквенных символов. Цифры и знаки исключены
# намеренно: организм учит РЕЧЬ, а не разметку.
WORD_PATTERN = re.compile(r"[^\s\d\W]+", flags=re.UNICODE)

# Гласные (русские и английские) для примитивного деления на слоги: слог —
# это согласные* + гласная(ые), а хвостовые согласные липнут к последнему
# слогу слова.
VOWELS: Set[str] = set("аеёиоуыэюяAEIOUYaeiouy")


class NeocortexMixin:
    """Словарь, слоги, удивление, понятия и схемы — одно вещество."""


    def process_language_input(
        self,
        text: str,
        timestamp: Optional[float] = None,
    ) -> LexicalProcessingResult:
        """
        A side process that does not block the main reply: it breaks
        incoming text into words and syllables, accumulating a primitive
        vocabulary regardless of whether the search hit or missed.

        For every word:
            1. A word node is upserted (node_type='word') — frequency
               raises its weight, which is what mastery means here.
            2. The word is split into syllables, each upserted as a
               syllable node (node_type='syllable').
            3. Each syllable is linked by an edge to its parent word
               (syllable_word_edge_weight).
        Neighbouring words in a sentence are joined by co-occurrence
        edges (word_cooccurrence_edge_weight) — a primitive grammar of
        adjacency.

        When lexical acquisition is disabled, this returns an empty result
        immediately, without touching the database.
        """
        if not self.settings.lexical_acquisition_enabled or not text or not text.strip():
            return LexicalProcessingResult(0, 0, 0, 0)

        ts = timestamp if timestamp is not None else time.time()

        # The same tokenisation as in compute_surprise — see the comment
        # there: the organism must be surprised by exactly the units it
        # learns.
        tokens = self._tokenize_for_lexicon(text)

        if not tokens:
            return LexicalProcessingResult(0, 0, 0, 0)

        words_processed = 0
        syllables_processed = 0
        new_words = 0
        new_syllables = 0
        previous_word_id: Optional[int] = None

        for token in tokens:
            word_id, word_was_created = self.db.upsert_lexical_node(
                node_type="word",
                text=token,
                initial_weight=self.settings.word_node_initial_weight,
                reinforce_step=self.settings.word_node_reinforce_step,
                timestamp=ts,
            )
            words_processed += 1
            new_words += 1 if word_was_created else 0

            for syllable in self._split_into_syllables(token):
                syllable_id, syll_was_created = self.db.upsert_lexical_node(
                    node_type="syllable",
                    text=syllable,
                    initial_weight=self.settings.syllable_node_initial_weight,
                    reinforce_step=self.settings.syllable_node_reinforce_step,
                    timestamp=ts,
                )
                syllables_processed += 1
                new_syllables += 1 if syll_was_created else 0

                self.connect_nodes(
                    syllable_id, word_id,
                    weight_boost=self.settings.syllable_word_edge_weight,
                    timestamp=ts,
                )

            if previous_word_id is not None:
                self.connect_nodes(
                    previous_word_id, word_id,
                    weight_boost=self.settings.word_cooccurrence_edge_weight,
                    timestamp=ts,
                )
            previous_word_id = word_id

        logger.debug(
            "[LEXICAL ACQUISITION] words=%d (new=%d) syllables=%d (new=%d) text=%r",
            words_processed, new_words, syllables_processed, new_syllables, text[:40],
        )

        return LexicalProcessingResult(
            words_processed=words_processed,
            syllables_processed=syllables_processed,
            new_words=new_words,
            new_syllables=new_syllables,
        )

    def compute_surprise(self, text: str) -> SurpriseResult:
        """
        Measures how UNEXPECTED an incoming text is for this particular
        organism, judged against the graph of language it has built up.

        This role used to be played by Shannon entropy over characters. It
        measured a property of the string and was completely blind to
        experience — an empty mind and one that had seen a phrase fifty
        times produced the same number. Four mechanisms depended on it:
        the spike gate, confidence, structural consolidation and
        curiosity. All four were steered by a quantity that learning never
        changed.

        Prediction error has two components:

            lexical     — are the WORDS THEMSELVES familiar? A word's
                          familiarity grows with its weight (frequency
                          equals mastery) and saturates at
                          vocabulary_mastery_min_weight.
            structural  — are the PAIRINGS of neighbouring words familiar?
                          A pair's familiarity grows with the weight of the
                          co-occurrence edge and saturates at
                          edge_activation_threshold.

        AN IMPORTANT LIMITATION: edges are stored undirected — upsert_edge
        normalises each pair by ascending id. The structural component
        therefore answers "have these words appeared next to each other",
        NOT "does one follow the other". This is not a language model and
        must not be called one; for prediction error that resolution is
        enough.

        Tokenisation deliberately matches process_language_input — the
        organism must be surprised by exactly the units it learns.

        Edge cases:
            empty text / no tokens -> 0.0 (nothing to be surprised by)
            a single token (no pairs) -> the lexical component only
            an empty graph -> 1.0 (everything is new to a newborn)
        """
        tokens = self._tokenize_for_lexicon(text)
        if not tokens:
            return SurpriseResult(0.0, 0.0, 0.0, 0, 0, 0, 0)

        # --- Lexical novelty: are the words themselves familiar? ---
        rows = self.db.get_lexical_nodes_by_texts("word", list(set(tokens)))
        known = {row["context"]: (row["id"], row["weight"]) for row in rows}

        mastery = max(1e-9, self.settings.vocabulary_mastery_min_weight)
        familiarities = [
            min(1.0, known[t][1] / mastery) if t in known else 0.0
            for t in tokens
        ]
        lexical_surprise = 1.0 - (sum(familiarities) / len(familiarities))

        # --- Structural novelty: are the pairings familiar? ---
        token_ids = [known[t][0] for t in tokens if t in known]
        edge_weights = {}
        for edge in self.db.get_edges_between(list(set(token_ids))):
            # Pairs are stored undirected, so both orders go into the
            # lookup and the actual word order of the input can be used.
            a, b, w = edge["node_from"], edge["node_to"], edge["weight"]
            edge_weights[(a, b)] = w
            edge_weights[(b, a)] = w

        activation = max(1e-9, self.settings.edge_activation_threshold)
        pair_familiarities: List[float] = []
        for left, right in zip(tokens, tokens[1:]):
            if left in known and right in known:
                weight = edge_weights.get((known[left][0], known[right][0]), 0.0)
                pair_familiarities.append(min(1.0, weight / activation))
            else:
                # If either word is unfamiliar, the pairing certainly is
                pair_familiarities.append(0.0)

        known_words = sum(1 for f in familiarities if f > 0.0)
        known_pairs = sum(1 for f in pair_familiarities if f > 0.0)

        if not pair_familiarities:
            # A single token carries no structural information at all, so
            # the result rests on lexis alone — renormalised rather than
            # crediting structural surprise with a fictitious 0 or 1.
            total = lexical_surprise
            structural_surprise = 0.0
        else:
            structural_surprise = 1.0 - (sum(pair_familiarities) / len(pair_familiarities))
            total = (
                self.settings.surprise_lexical_weight * lexical_surprise
                + self.settings.surprise_structural_weight * structural_surprise
            )
            weight_sum = self.settings.surprise_lexical_weight + self.settings.surprise_structural_weight
            if weight_sum > 0:
                total /= weight_sum

        # --- Correction for the AMOUNT of content ---
        #
        # Surprise was the mean unfamiliarity, and a mean is blind to HOW
        # MUCH content an utterance carries: on empty memory "uh-huh" and
        # "my daughter Lisa is six" both scored 1.000. For a growing bot
        # that hardly mattered — it babbles either way — but it cluttered
        # the library with interjections from day one: a run over a stream
        # resembling a real assistant conversation stored "thanks", "okay"
        # and "uh-huh" alongside the penicillin allergy.
        #
        # One unfamiliar word carries less information than six — that is
        # a definition, not a heuristic. So surprise is multiplied by the
        # fraction of content gathered, saturating at
        # surprise_full_content_tokens: a short utterance physically
        # cannot surprise much, however novel it may be.
        full = max(1, self.settings.surprise_full_content_tokens)
        total *= min(1.0, len(tokens) / full)

        total = max(0.0, min(1.0, total))

        logger.debug(
            "[SURPRISE] text=%r total=%.3f (lex=%.3f structural=%.3f) "
            "known_words=%d/%d known_pairs=%d/%d",
            text[:40], total, lexical_surprise, structural_surprise,
            known_words, len(tokens), known_pairs, len(pair_familiarities),
        )

        return SurpriseResult(
            total=total,
            lexical=lexical_surprise,
            structural=structural_surprise,
            known_words=known_words,
            total_words=len(tokens),
            known_pairs=known_pairs,
            total_pairs=len(pair_familiarities),
        )

    def _tokenize_for_lexicon(self, text: str) -> List[str]:
        """
        One tokenisation for the whole lexical layer, shared by learning
        (process_language_input) and by surprise (compute_surprise): the
        organism must be surprised by exactly the units it goes on to
        memorise, or the measurement is of something other than what is
        being learned.
        """
        if not text or not text.strip():
            return []
        return [
            w.lower() for w in WORD_PATTERN.findall(text)
            if len(w) >= self.settings.lexical_min_token_length
        ][: self.settings.lexical_max_tokens_per_input]

    @staticmethod
    def _split_into_syllables(word: str) -> List[str]:
        """
        Primitive syllable segmentation: a syllable accumulates character
        by character and closes on the first vowel; trailing consonants at
        the end of a word stick to the last syllable found. It makes no
        claim to linguistic accuracy — it is enough for babbling.
        """
        syllables: List[str] = []
        current = ""

        for ch in word:
            current += ch
            if ch in VOWELS:
                syllables.append(current)
                current = ""

        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)

        return syllables if syllables else [word]

    def get_vocabulary_size(self) -> int:
        """
        The number of CONSOLIDATED words — word nodes with
        weight >= vocabulary_mastery_min_weight. A word heard once creates
        a node with a low initial weight and does NOT count here until the
        user has repeated it a few more times. Used to gate the stage of
        speech development and for the user-facing status report, so it
        reflects what the bot has genuinely LEARNED rather than everything
        that ever passed through it.
        """
        return self.db.count_mastered_words(self.settings.vocabulary_mastery_min_weight)

    def get_exposed_vocabulary_size(self) -> int:
        """
        The total number of DISTINCT words the bot has heard at least
        once, consolidated or not — the passive vocabulary. For statistics
        and debugging only: the gap between this and get_vocabulary_size()
        shows how many words are still on their way to being learned. NOT
        used to gate speech stages.
        """
        return self.gate.semantic.count_words()

    def get_mastered_words_in(self, text: str) -> List[KnownWord]:
        """
        Which words of an INCOMING message the organism has actually
        mastered, in the order they appear in the text.

        This is the first place where the learned vocabulary affects WHAT
        the bot says, rather than only the counter that permits it to
        speak. Knowing a word used to exist purely as a number: the bot
        could know "hello" better than any other word it had (weight
        0.747) and still answer a greeting with random syllables, because
        all that reached generation was the size of the vocabulary.

        A word counts as mastered at weight >=
        vocabulary_mastery_min_weight — the same bar as in
        get_vocabulary_size, or the bot would utter words it does not
        itself consider learned.
        """
        return self._words_in(text, mastered=True)

    def get_emerging_words_in(self, text: str) -> List[KnownWord]:
        """
        Words of the incoming phrase the organism HAS HEARD but has NOT
        yet MASTERED — its zone of proximal development.

        These are the candidates for EXPLORATION. Until this existed the
        architecture had no mechanism at all for trying the unmastered:
        the organism only exploited what it already knew and turned
        everything else into babble. A purely exploiting system does not
        develop — it converges and freezes.

        These words specifically, rather than wholly unfamiliar ones:
        attempting something far beyond current competence is pointless —
        the attempt fails and teaches nothing. Learning happens at the
        edge of what is already known.
        """
        return self._words_in(text, mastered=False)

    def _words_in(self, text: str, mastered: bool) -> List[KnownWord]:
        """
        Shared selection of words from an incoming phrase by the mastery
        threshold.
        mastered=True  -> weight >= vocabulary_mastery_min_weight (its own words)
        mastered=False -> weight below it (heard, but not yet consolidated)
        """
        tokens = self._tokenize_for_lexicon(text)
        if not tokens:
            return []

        threshold = self.settings.vocabulary_mastery_min_weight
        rows = self.db.get_lexical_nodes_by_texts("word", list(set(tokens)))
        known = {
            row["context"]: (row["id"], row["weight"], row["reward_expectation"] or 0.0)
            for row in rows
            if (row["weight"] >= threshold) is mastered
        }

        result: List[KnownWord] = []
        seen: Set[str] = set()
        for token in tokens:
            if token in known and token not in seen:
                seen.add(token)
                node_id, weight, expectation = known[token]
                result.append(
                    KnownWord(
                        id=node_id,
                        text=token,
                        weight=weight,
                        reward_expectation=expectation,
                        # Preference = mastery plus a leaning towards what
                        # earned praise. Weight remains the main criterion,
                        # or the organism would start speaking in rare
                        # words that were praised once instead of the ones
                        # it actually commands.
                        preference=weight + expectation * self.settings.reward_preference_weight,
                    )
                )
        return result

    def get_top_words(self, limit: int = 8) -> List["tuple[str, float]"]:
        """
        The best-mastered words (text, weight) — for a status report.
        Shows the teacher what has actually taken hold in the bot's
        language.
        """
        rows = self.gate.semantic.top_words(limit)
        return [(row["context"], row["weight"]) for row in rows]

    def get_known_syllables(self, limit: Optional[int] = None) -> List[KnownSyllable]:
        """
        A pool of known syllables (id, text, weight) — candidates for the
        WEIGHTED choice made by the babbling subsystem. The pool itself is
        sampled at random, but every element carries its real weight; the
        weighted draw happens in the caller, not here.

        `limit` defaults to settings.babbling_syllable_pool_size.
        """
        effective_limit = limit if limit is not None else self.settings.babbling_syllable_pool_size
        rows = self.db.get_random_nodes_by_type("syllable", limit=effective_limit)
        return [
            KnownSyllable(id=row["id"], text=row["context"], weight=row["weight"])
            for row in rows
        ]

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

    def find_hub_clusters(
        self,
        min_edge_weight: Optional[float] = None,
        min_spokes: Optional[float] = None,
        max_spokes: Optional[float] = None,
        limit: int = 1,
        timestamp: Optional[float] = None,
    ) -> List[HubCluster]:
        """
        Looks for hub-and-spoke clusters: a dominant node with the
        largest sum of strong edge weights (its hub score) and its top-N
        strongest neighbours, as candidates for semantic consolidation
        during the sleep phase.

        Returns up to `limit` clusters sorted by hub score. A cluster is
        included only when the hub has at least min_spokes strong
        neighbours — otherwise it is not a cluster but merely a pair.
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.sleep_hub_min_edge_weight
        )
        effective_min_spokes = (
            min_spokes if min_spokes is not None else self.settings.sleep_min_cluster_spokes
        )
        effective_max_spokes = (
            max_spokes if max_spokes is not None else self.settings.sleep_max_cluster_spokes
        )

        hub_rows = self.db.get_hub_candidates(min_edge_weight=effective_edge_weight)

        clusters: List[HubCluster] = []
        used_node_ids: set = set()

        for hub_row in hub_rows:
            if len(clusters) >= limit:
                break

            hub_id = hub_row["id"]
            if hub_id in used_node_ids:
                continue

            associated = self.get_associated_nodes(
                hub_id,
                min_weight=effective_edge_weight,
                limit=effective_max_spokes,
                timestamp=timestamp,
            )

            # Drop spokes already claimed by another cluster in this same
            # pass, so no node is consolidated twice
            available_spokes = [a for a in associated if a.id not in used_node_ids]

            if len(available_spokes) < effective_min_spokes:
                continue

            cluster = HubCluster(
                hub_id=hub_id,
                hub_context=hub_row["context"],
                hub_response=hub_row["response"],
                hub_weight=hub_row["weight"],
                spoke_ids=[a.id for a in available_spokes],
                spoke_contexts=[a.context for a in available_spokes],
                spoke_responses=[a.response for a in available_spokes],
                spoke_weights=[a.weight for a in available_spokes],
                edge_weights=[a.edge_weight for a in available_spokes],
            )

            clusters.append(cluster)
            used_node_ids.add(hub_id)
            used_node_ids.update(cluster.spoke_ids)

            logger.info(
                "[SLEEP CLUSTER] Cluster found: hub=%s (weight=%.2f) + spokes=%s",
                hub_id, hub_row["weight"], cluster.spoke_ids,
            )

        return clusters

    def create_abstract_node(
        self,
        summary_context: str,
        summary_response: str,
        source_node_ids: List[int],
        weight: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Stores a new abstract long-term node — the result of semantically
        consolidating a cluster — and ARCHIVES the cluster's source nodes
        by multiplying their weight by sleep_archive_weight_multiplier.
        They are not deleted immediately: they become candidates for
        deletion at the next ordinary decay or pruning pass if they remain
        unused.

        The new abstract node is also linked by edges to every source node
        of the cluster, preserving a trace of where it came from.

        Returns the id of the new abstract node.
        """
        effective_weight = weight if weight is not None else self.settings.sleep_abstract_node_weight
        ts = timestamp if timestamp is not None else time.time()

        # ТОТ ЖЕ ТИП, ЧТО У СВЁРТОК ЭПИЗОДА, и по той же измеренной
        # причине: абстракция, пущенная в общий поиск, содержит слишком
        # много слов и совпадает почти с любым запросом. У консолидации
        # это роняло R@1 с 76% до 42%, а понижение силы спасало лишь
        # частично (56% и 60%), потому что выигрывала она НА
        # РЕЛЕВАНТНОСТИ, а не на важности.
        #
        # Схема достаётся отдельным ходом — Memory.summaries().
        abstract_node_id = self.db.insert_node(
            context=summary_context,
            response=summary_response,
            weight=effective_weight,
            timestamp=ts,
            node_type="episode_summary",
        )

        for source_id in source_node_ids:
            source_row = self.db.get_node(source_id)
            if source_row is None:
                continue

            archived_weight = source_row["weight"] * self.settings.sleep_archive_weight_multiplier
            self.db.update_weight(source_id, archived_weight)

            # АБСТРАКЦИЯ ЗАБИРАЕТ ДОЛЮ, а не прибавляется сверху. Прежде
            # сон добавлял узел и не убирал ни одного, то есть память
            # росла монотонно с каждым циклом. В модели интерференции
            # ценность — доля от суммы, поэтому источники обязаны отдать
            # ровно столько, сколько получила свёртка.
            if self.settings.use_relative_strength:
                base = source_row["strength"]
                if base is None:
                    base = source_row["weight"]
                self.db.add_strength(
                    source_id,
                    base * (self.settings.sleep_archive_weight_multiplier - 1.0),
                    self.settings.strength_max,
                )

            self.connect_nodes(abstract_node_id, source_id, weight_boost=self.settings.edge_initial_weight, timestamp=ts)

        logger.info(
            "[SLEEP CONSOLIDATION] Abstract node id=%s (weight=%.2f) created from cluster %s",
            abstract_node_id, effective_weight, source_node_ids,
        )

        return abstract_node_id
