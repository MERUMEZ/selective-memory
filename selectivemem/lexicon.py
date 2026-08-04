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
 LEXICON.PY — Словарь и собственное удивление
================================================================================
Здесь живёт то, что в мозге принадлежит КОРЕ, а не гиппокампу: знание слов
и слогов, накопленное из всего виденного, и построенная на нём оценка
новизны.

ПОЧЕМУ ЭТО ВЫДЕЛЕНО ПЕРВЫМ. Лексика лежала в одном файле с эпизодической
памятью и в одной таблице с ней — узлы word и syllable рядом с
воспоминаниями. Это не только путаница в названиях: подсчёт узлов включал
словарь, порог сна пробивался на девятом сообщении, и сон запускался на
каждое следующее. Кратковременная память очищалась ежесообщение, а в
проде уходило по два вызова языковой модели на реплику. Починили тогда
заплаткой в count_memory_nodes, а причина была в том, что корковое знание
считалось эпизодами.

ЧТО ЗДЕСЬ ГЛАВНОЕ — compute_surprise. Организм не берёт новизну из
языковой модели, он меряет её ПО СВОЕМУ ОПЫТУ: доля незнакомых слов и
доля незнакомых пар соседей. От этого числа зависит вся запись.

ИЗВЕСТНЫЙ ПРЕДЕЛ, записанный здесь честно: удивление считается по словам
и парам слов, то есть слепо к смыслу. Перефразировка знакомой мысли
удивляет сильнее, чем должна. Лечится это семантическим удивлением —
сравнением входа с ближайшими соседями по вектору, — и это отдельная
работа.

Класс — миксин: состояние (db, settings) принадлежит MemoryGraph, который
его подмешивает. Разделение пока файловое, а не по владению данными;
переход на composition — следующий шаг, и он ломает вызовы, поэтому
делается отдельно и с замером.
================================================================================
"""

import logging
import re
from typing import List, Optional, Set

from selectivemem.records import (
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


class LexiconMixin:
    """Словарь, слоги и собственное удивление организма."""

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
        return self.db.count_nodes_by_type("word")

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
        rows = self.db.get_top_nodes_by_type("word", limit=limit)
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
