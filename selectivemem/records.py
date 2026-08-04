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
 RECORDS.PY — Что память возвращает наружу
================================================================================
Двенадцать структур, которыми память отвечает на вопросы: найденное
совпадение, результат свёртки, разложение удивления, отчёт о подрезке.

ПОЧЕМУ ОТДЕЛЬНЫМ ФАЙЛОМ. Они лежали в graph_memory.py, и это мешало
разнести сам граф по участкам: любой выделенный модуль импортировал бы
graph_memory ради MemoryMatch, а graph_memory импортировал бы его в
ответ — круг. Здесь у структур нет зависимостей вовсе, поэтому импортировать
их может кто угодно.

Имена по-прежнему доступны из selectivemem.graph_memory: снаружи их
импортируют тесты, стенды и витрина, и ломать это ради перестановки
файлов незачем.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


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
