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
 GRAPH_MEMORY.PY — The biological layer of long-term memory
================================================================================
MemoryGraph sits on top of Database and adds the living logic:
    - storing a new link with an initial weight (spike memory)
    - finding similar context by keywords plus fuzzy similarity (search)
    - updating last_accessed on every touch
    - weight decay for old, unused links
    - SELECTIVE CONSOLIDATION — moving episodes from short-term memory
      (WorkingMemory) into long-term storage

Consolidation (consolidate_from_stm) judges an accumulated STM episode by
two criteria and reaches one of three decisions:
    a) EMOTIONAL NODE — high emotion_score/density -> stored with a high
       weight (the analogue of the spike gate, but for a whole episode).
    b) STRUCTURAL NODE — high average surprise (a lot of new information)
       -> stored with a moderate weight.
    c) ROUTINE NOISE — neither of the above -> the episode is discarded
       without being stored at all.

The decay formula (an exponential forgetting curve):
    weight(t) = weight_0 * exp(-DECAY_RATE * dt / AGE_T0)
================================================================================
"""

import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, List, Optional, Set, TYPE_CHECKING

from selectivemem.database import Database
from selectivemem.settings import MemorySettings
from selectivemem import embeddings
import logging

if TYPE_CHECKING:
    from selectivemem.working_memory import STMEntry

logger = logging.getLogger(__name__)

# Stop words for keyword matching. Russian entries are here because the
# bundled encoder is Russian; English ones because that is the other
# language this was tested in. With your own encoder and language, extend
# this set — it only filters noise from the lexical component of scoring.
STOP_WORDS: Set[str] = {
    "и", "в", "на", "с", "по", "к", "у", "из", "за", "от", "до", "для",
    "что", "как", "это", "то", "я", "ты", "он", "она", "мы", "вы", "они",
    "не", "но", "а", "же", "бы", "ли", "или", "тот", "его", "ее", "их",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "and", "or", "but", "for", "with", "at", "by", "it", "this", "that",
}

WORD_PATTERN = re.compile(r"[^\s\d\W]+", flags=re.UNICODE)

# Vowels (Russian + English) for primitive syllable segmentation: a
# syllable is consonants* + vowel(s), and trailing consonants stick to the
# last syllable of the word. Add your own alphabet's vowels here if the
# babbling showcase matters to you; the memory itself does not use this.
VOWELS: Set[str] = set("аеёиоуыэюяAEIOUYaeiouy")


@dataclass
class LexicalProcessingResult:
    """Result of the side process of lexical language acquisition."""
    words_processed: int
    syllables_processed: int
    new_words: int
    new_syllables: int


@dataclass
class SurpriseResult:
    """
The organism's prediction error on incoming text — its OWN surprise,
    computed against its own graph rather than a statistic of the string.

    total      — the result [0..1], fed to the spike gate, confidence and
                 consolidation
    lexical    — the share of novelty coming from unfamiliar WORDS
    structural — the share coming from unfamiliar PAIRINGS of neighbours
    known_words / total_words — how many of the input's words are known
    known_pairs / total_pairs — the same for pairs of neighbours
    """
    total: float
    lexical: float
    structural: float
    known_words: int
    total_words: int
    known_pairs: int
    total_pairs: int


@dataclass
class MemoryMatch:
    """A match found while searching the memory graph for similar context."""
    id: int
    context: str
    response: str
    weight: float
    similarity: float
    created_at: float
    last_accessed: float


@dataclass
class ConsolidationResult:
    """Outcome of trying to consolidate an STM episode into long-term memory."""
    decision: str  # "emotional_node" | "structural_node" | "routine_noise"
    node_id: Optional[int] = None
    weight: Optional[float] = None
    reason: str = ""


@dataclass
class KnownSyllable:
    """
    A known syllable with its id and current weight (mastery) — used by
    the babbling subsystem for weighted syllable choice and for tracking
    which nodes were used, so reinforcement can reach them.
    """
    id: int
    text: str
    weight: float


@dataclass
class RewardSignal:
    """
    The dopamine signal for one node (see MemoryGraph.apply_reward).

    prediction_error is the "praised unexpectedly" quantity: it, rather
    than raw valence, drives both the rate of consolidation and the bias
    of future choices.
    """
    node_id: int
    valence: float
    expected: float
    prediction_error: float
    new_expectation: float


@dataclass
class SupersededNode:
    """
    A memory superseded by a newer version of the same fact (see
    MemoryGraph.find_superseded).

    word_overlap is kept for debugging: it shows why a node was judged a
    different version rather than a repetition.
    """
    id: int
    context: str
    similarity: float
    word_overlap: float


@dataclass
class KnownWord:
    """
    A mastered word found in an incoming message (see
    MemoryGraph.get_mastered_words_in). The id is there so reinforcement
    can strengthen exactly the WORDS the bot used successfully — the loop
    used to receive only babbling syllables.
    """
    id: int
    text: str
    weight: float
    reward_expectation: float = 0.0
    # What the organism actually chooses to say by: mastery plus a
    # leaning towards what earned praise (see get_mastered_words_in)
    preference: float = 0.0


@dataclass
class AssociatedNode:
    """A node pulled in by spreading activation along an associative edge."""
    id: int
    context: str
    response: str
    weight: float
    edge_weight: float
    activation_score: float
    source_node_id: int


@dataclass
class ActivationTrace:
    """One associative activation: node A caused node B to be pulled in."""
    source_id: int
    target_id: int
    edge_weight: float
    activation_score: float




@dataclass
class HubCluster:
    """
    A hub-and-spoke cluster: a dominant node (hub) and its strongly
    linked neighbours (spokes), candidates for semantic consolidation
    during the sleep phase.
    """
    hub_id: int
    hub_context: str
    hub_response: str
    hub_weight: float
    spoke_ids: List[int]
    spoke_contexts: List[str]
    spoke_responses: List[str]
    spoke_weights: List[float]
    edge_weights: List[float]


@dataclass
class PruningReport:
    """Result of synaptic pruning and edge cleaning."""
    edges_pruned: int
    orphan_nodes_pruned: int




class MemoryGraph:
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

    # ----------------------------------------------------------------------
    # SURPRISE — the organism's own prediction error
    # ----------------------------------------------------------------------

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
        found: List[SupersededNode] = []

        for row in self.db.fetch_searchable_nodes():
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

            found.append(
                SupersededNode(
                    id=row["id"],
                    context=row["context"],
                    similarity=similarity,
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
        )

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
            try:
                import numpy as np

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
            except ImportError:
                pass

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
                "pip install selective-memory[semantic]"
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

        for row, keyword_score, semantic_score in prefiltered:
            context_normalized = row["context"].strip().lower()
            fuzzy_score = self._compute_fuzzy_similarity(query_normalized, context_normalized)

            combined_score = (
                keyword_score * self.settings.memory_keyword_weight
                + fuzzy_score * self.settings.memory_fuzzy_weight
                + semantic_score * self.settings.memory_semantic_weight
                + row["weight"] * self.settings.memory_weight_influence
            )
            combined_score = min(1.0, combined_score)

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
        top_matches = scored[:top_k]

        for match in top_matches:
            self.touch_node(match.id, timestamp=timestamp)

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
    def _keyword_overlap(query_keywords: Set[str], context_keywords: Set[str]) -> float:
        if not query_keywords or not context_keywords:
            return 0.0
        intersection = query_keywords & context_keywords
        smallest_set_size = min(len(query_keywords), len(context_keywords))
        return len(intersection) / smallest_set_size if smallest_set_size else 0.0

    @staticmethod
    def _compute_fuzzy_similarity(query: str, context: str) -> float:
        return SequenceMatcher(None, query, context).ratio()

    # ----------------------------------------------------------------------
    # 3. Updating last_accessed on a touch
    # ----------------------------------------------------------------------

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

        return results

    # ----------------------------------------------------------------------
    # 4. Decay — the fading of old links
    # ----------------------------------------------------------------------

    def apply_decay(self, now: Optional[float] = None) -> int:
        """
        Applies exponential weight decay to every long-term node and to
        the edges of the associative graph. Rare or unused edges (weight
        below edge_forget_threshold) are deleted outright.
        """
        current_time = now if now is not None else time.time()

        decayed_count = self._decay_nodes(current_time)
        self._decay_edges(current_time)

        return decayed_count

    # Lexical nodes are the infrastructure of language rather than
    # episodes of a conversation, so they live on their own, far longer
    # timescale.
    LEXICAL_NODE_TYPES = frozenset({"word", "syllable"})

    def _age_t0_for(self, node_type: Optional[str]) -> float:
        """
        The characteristic lifetime of a node for the decay formula.
        Vocabulary fades on lexical_age_t0 (~30 days), everything else on
        age_t0 (~7 subjective hours).

        Without that split, a mastered word lost its status overnight and
        was deleted from the database within a day — the vocabulary could
        never accumulate at all.
        """
        if node_type in MemoryGraph.LEXICAL_NODE_TYPES:
            return self.settings.lexical_age_t0
        return self.settings.age_t0

    def _decay_nodes(self, current_time: float) -> int:
        """Exponential weight decay for nodes."""
        rows = self.db.fetch_all_nodes()

        updates = []
        to_forget = []
        decayed_count = 0
        skipped_meta_count = 0

        for row in rows:
            if row["is_meta"]:
                skipped_meta_count += 1
                continue

            # NULL guard: if last_decayed_at was never set — some path
            # of creation or update missed it — last_accessed serves as
            # the origin instead of crashing with a TypeError. The same
            # pass fills last_decayed_at in through updates[] below.
            last_decayed = row["last_decayed_at"]
            if last_decayed is None:
                last_decayed = row["last_accessed"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = row["weight"]
            # Effective lifetime = the base for this node type times
            # stability. Stability grows on every recall (see
            # Database.update_last_accessed), so memory that is in demand
            # resists time while memory that is not disappears quickly.
            stability = row["stability"] if row["stability"] else self.settings.stability_initial
            effective_t0 = self._age_t0_for(row["node_type"]) * max(1e-9, stability)
            decay_factor = math.exp(-self.settings.decay_rate * dt / effective_t0)

            # THE DECAY FLOOR. A node the user praised dims but does not
            # vanish: it decays towards a non-zero level rather than to
            # nothing.
            #
            # Why. Stability grows through RECALL, so something marked as
            # important but never needed again still sank to zero and was
            # deleted. Measured: after two weeks of silence, NOTHING
            # survived a fifteen-message conversation — including what the
            # user had explicitly called important. For an assistant that
            # was told "I am allergic to penicillin" that is unacceptable:
            # such a thing must never be forgotten, even if it never came
            # up again.
            #
            # Where the floor's height comes from: reward_expectation, the
            # very quantity the Rescorla-Wagner rule maintains (see
            # apply_reward). It already means "how much approval this node
            # has earned" — a ready-made answer to "how important is this
            # TO THE USER". No extra column is needed and no threshold is
            # assigned by hand.
            #
            # A node that was never reinforced has an expectation of 0 and
            # a floor of 0 — nothing changes for it, and it fades as
            # before. So this does not make memory immortal: only a
            # handful of nodes ever get a floor.
            expectation = row["reward_expectation"] or 0.0
            floor = max(
                # Earned by approval: the more a node was reinforced, the
                # higher it settles.
                max(0.0, expectation) * self.settings.memory_floor_max,
                # Earned by being written at all. Measured: without this,
                # decay DELETED every evidence node for knowledge-update
                # questions — 12 of 12, five instances out of five — while
                # only a tenth of memory was removed overall. The evidence
                # is simply the oldest thing in the store, and age alone
                # decided its fate.
                self.settings.memory_floor_base,
                # Earned by the force of the event. A flat floor saves
                # everything and so forgets nothing, which costs the whole
                # praised-over-routine gap; this term keeps the gap, because
                # routine that barely cleared the gate gets a floor below
                # forget_threshold and still dies of old age.
                (row["spike_strength"] or 0.0) * self.settings.memory_floor_spike_factor,
            )
            floor = min(floor, old_weight)          # the floor never raises a weight
            new_weight = floor + (old_weight - floor) * decay_factor

            if new_weight < self.settings.forget_threshold:
                to_forget.append(row["id"])
            else:
                updates.append({
                    "id": row["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_weights(updates)
            logger.info(
                "[DECAY APPLIED] Weights updated: %d nodes (meta skipped: %d)",
                len(updates), skipped_meta_count,
            )

        for node_id in to_forget:
            self.db.delete_node(node_id)

        if to_forget:
            logger.info("[MEMORY FORGOTTEN] Nodes deleted (weight < FORGET_THRESHOLD): %d", len(to_forget))

        return decayed_count

    def _decay_edges(self, current_time: float) -> int:
        """
        Exponential weight decay for edges. Edges fade faster than nodes
        (edge_decay_rate is normally above decay_rate), modelling the fact
        that associations between memories are more fragile than the
        memories themselves. Edges below edge_forget_threshold are deleted
        outright.
        """
        edges = self.db.fetch_all_edges()

        updates = []
        to_forget = []
        decayed_count = 0

        for edge in edges:
            # NULL guard — see the comment in _decay_nodes.
            last_decayed = edge["last_decayed_at"]
            if last_decayed is None:
                last_decayed = edge["last_activated"]

            dt = current_time - last_decayed
            if dt <= 0:
                continue

            old_weight = edge["weight"]
            decay_factor = math.exp(-self.settings.edge_decay_rate * dt / self.settings.age_t0)
            new_weight = old_weight * decay_factor

            if new_weight < self.settings.edge_forget_threshold:
                to_forget.append(edge["id"])
            else:
                updates.append({
                    "id": edge["id"],
                    "weight": new_weight,
                    "last_decayed_at": current_time,
                })
                decayed_count += 1

        if updates:
            self.db.bulk_update_edge_weights(updates)
            logger.info(
                "[EDGE DECAY APPLIED] Weights updated: %d edges (t=%.2f)",
                len(updates), current_time,
            )

        for edge_id in to_forget:
            self.db.delete_edge(edge_id)

        if to_forget:
            logger.info(
                "[EDGE FORGOTTEN] Edges deleted (weight < EDGE_FORGET_THRESHOLD): %d",
                len(to_forget),
            )

        return decayed_count

    # ----------------------------------------------------------------------
    # 4b. SLEEP CYCLE — synaptic pruning and edge cleaning
    # ----------------------------------------------------------------------

    def prune_weak_edges(self, min_weight: Optional[float] = None) -> int:
        """
        Explicit, non-decay pruning of edges: deletes EVERY edge with
        weight < min_weight. Used by the sleep phase, unlike _decay_edges,
        which first lowers weights and deletes only what fell below the
        threshold DURING that call.

        Returns the number of edges deleted.
        """
        threshold = min_weight if min_weight is not None else self.settings.edge_forget_threshold
        deleted = self.db.delete_edges_below_weight(threshold)
        return deleted

    def prune_orphan_nodes(
        self,
        min_edge_weight: Optional[float] = None,
        max_node_weight: Optional[float] = None,
    ) -> int:
        """
        Deletes orphaned nodes: those with no strong edge (weight >=
        min_edge_weight) AND weak in themselves (node weight below
        max_node_weight). Strong isolated memories are left alone — they
        stand on their own even when linked to nothing.

        Returns the number of nodes deleted.
        """
        effective_edge_weight = (
            min_edge_weight if min_edge_weight is not None else self.settings.edge_activation_threshold
        )
        effective_node_weight = (
            max_node_weight if max_node_weight is not None else self.settings.sleep_orphan_weight_threshold
        )

        orphans = self.db.get_orphan_nodes(
            min_edge_weight=effective_edge_weight,
            max_node_weight=effective_node_weight,
        )

        for orphan in orphans:
            self.db.delete_node(orphan["id"])

        if orphans:
            logger.info(
                "[SLEEP PRUNING] Orphan nodes deleted (weight < %.2f, no edges >= %.2f): %d",
                effective_node_weight, effective_edge_weight, len(orphans),
            )

        return len(orphans)

    def run_synaptic_pruning(self) -> PruningReport:
        """
        The full synaptic pruning cycle: weak edges are cut first, orphan
        nodes are sought AFTERWARDS. The order matters — removing weak
        edges can orphan nodes that were held up by nothing else.
        """
        edges_pruned = self.prune_weak_edges()
        orphan_nodes_pruned = self.prune_orphan_nodes()

        return PruningReport(
            edges_pruned=edges_pruned,
            orphan_nodes_pruned=orphan_nodes_pruned,
        )

    # ----------------------------------------------------------------------
    # 4c. SLEEP CYCLE — hub-and-spoke clustering
    # ----------------------------------------------------------------------

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

        abstract_node_id = self.db.insert_node(
            context=summary_context,
            response=summary_response,
            weight=effective_weight,
            timestamp=ts,
        )

        for source_id in source_node_ids:
            source_row = self.db.get_node(source_id)
            if source_row is None:
                continue

            archived_weight = source_row["weight"] * self.settings.sleep_archive_weight_multiplier
            self.db.update_weight(source_id, archived_weight)

            self.connect_nodes(abstract_node_id, source_id, weight_boost=self.settings.edge_initial_weight, timestamp=ts)

        logger.info(
            "[SLEEP CONSOLIDATION] Abstract node id=%s (weight=%.2f) created from cluster %s",
            abstract_node_id, effective_weight, source_node_ids,
        )

        return abstract_node_id

        # ----------------------------------------------------------------------
    # 4d. PROACTIVE MEMORY RECALL — choosing a node to speak up about
    # ----------------------------------------------------------------------



    # ----------------------------------------------------------------------
    # 5. SELECTIVE CONSOLIDATION (short-term -> long-term)
    # ----------------------------------------------------------------------

    def consolidate_from_stm(
        self,
        entries: List["STMEntry"],
        timestamp: Optional[float] = None,
        already_captured_by_spike: bool = False,
    ) -> ConsolidationResult:
        """
        Judges an accumulated short-term episode and decides according to
        the physics of selective consolidation:

            a) EMOTIONAL NODE — if the highest emotion_score among the STM
               entries reaches stm_emotional_threshold, the episode is
               packed and stored with a high weight (that same maximum).

            b) STRUCTURAL NODE — if the episode's average surprise reaches
               stm_structural_threshold (a lot of new information), it is
               packed and stored with a moderate weight
               (stm_structural_weight).

            c) ROUTINE NOISE — neither of the above: the episode is simply
               discarded without touching the database, which is how
               memory is spared everyday chatter.

        `entries` is a list of STMEntry, normally the result of
        working_memory.consume_all(). An empty list counts as routine
        noise straight away — there is no data.

        Returns a ConsolidationResult with the decision and the details
        for logs and debugging.
        """
        if not entries:
            return ConsolidationResult(decision="routine_noise", reason="STM is empty, no data")

        # Duplicate guard: if this flush was triggered by a spike that
        # already created an exact long-term node for the current
        # exchange, and STM at flush time holds only that same exchange
        # (at most two entries — one user turn and one bot turn), then
        # consolidation would create a near-identical duplicate. The write
        # is skipped: the content is already stored.
        if already_captured_by_spike and len(entries) <= 2:
            logger.info(
                "[CONSOLIDATION] Skipped: the exchange is already stored as a spike node "
                "(entries=%d)",
                len(entries),
            )
            return ConsolidationResult(
                decision="routine_noise",
                reason="Already stored as a spike node, duplicate skipped",
            )

        max_emotion = max(e.emotion_score for e in entries)
        avg_perplexity = sum(e.perplexity for e in entries) / len(entries)

        packed_context, packed_response = self._pack_episode(entries)

        # --- a) Emotional node (takes priority over the structural one) ---
        if max_emotion >= self.settings.stm_emotional_threshold:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=max_emotion,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Emotional node id=%s weight=%.3f (max_emotion=%.3f)",
                node_id, max_emotion, max_emotion,
            )
            return ConsolidationResult(
                decision="emotional_node",
                node_id=node_id,
                weight=max_emotion,
                reason=f"max_emotion={max_emotion:.3f} >= {self.settings.stm_emotional_threshold}",
            )

        # --- b) Structural node ---
        if avg_perplexity >= self.settings.stm_structural_threshold:
            node_id = self.save_connection(
                context=packed_context,
                response=packed_response,
                weight=self.settings.stm_structural_weight,
                timestamp=timestamp,
            )
            logger.info(
                "[CONSOLIDATION] Structural node id=%s weight=%.3f (avg_perplexity=%.3f)",
                node_id, self.settings.stm_structural_weight, avg_perplexity,
            )
            return ConsolidationResult(
                decision="structural_node",
                node_id=node_id,
                weight=self.settings.stm_structural_weight,
                reason=f"avg_perplexity={avg_perplexity:.3f} >= {self.settings.stm_structural_threshold}",
            )

        # --- c) Routine noise — discarded without touching the database ---
        logger.info(
            "[STM FLUSH] Routine noise discarded (max_emotion=%.3f, avg_perplexity=%.3f, %d entries)",
            max_emotion, avg_perplexity, len(entries),
        )
        return ConsolidationResult(
            decision="routine_noise",
            reason=f"max_emotion={max_emotion:.3f}, avg_surprise={avg_perplexity:.3f} — below both thresholds",
        )

    @staticmethod
    def _pack_episode(entries: List["STMEntry"]) -> "tuple[str, str]":
        """
        Packs a list of STMEntry into a (context, response) pair for
        storage as one long-term node. User turns are joined into context
        and bot turns into response, preserving their order.
        """
        user_lines = [e.text.strip() for e in entries if e.role == "user"]
        bot_lines = [e.text.strip() for e in entries if e.role != "user"]

        context = " | ".join(user_lines) if user_lines else "(no user turns)"
        response = " | ".join(bot_lines) if bot_lines else "(no bot replies)"

        return context, response

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