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
 SETTINGS.PY — Settings of the memory core
================================================================================
The third step of extracting the package: the core NO LONGER READS the
application's global config.

memory/* used to import config.py — a thousand-line module holding a
Telegram token, prompts and persona settings. Unacceptable for a library
installed via pip: it must carry its own configuration rather than depend
on one particular application's file.

The defaults are the values calibrated by measurement (see AUDIT.md). An
application may pass its own:

    graph = MemoryGraph(db=db, settings=MemorySettings(decay_rate=0.02))

or build them from its config wholesale:

    graph = MemoryGraph(db=db, settings=MemorySettings.from_module(config))

This file was GENERATED from config.py so the values are guaranteed to
match: retyping seventy constants by hand is a reliable way to change
something without noticing. test_settings.py checks them on every run.
================================================================================
"""

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class MemorySettings:
    """Memory core parameters. Names mirror config.py, lower-cased."""

    age_t0: float = 25200.0
    babbling_syllable_pool_size: int = 30
    base_plasticity_threshold: float = 0.25
    concept_max_similar_links: int = 2
    concept_node_weight: float = 0.7
    concept_similarity_edge_weight: float = 0.25
    concept_similarity_link_threshold: float = 0.35
    concept_user_edge_weight: float = 0.4
    contradiction_correction_relief: float = 0.15
    contradiction_repeat_threshold: float = 0.85
    contradiction_stability_factor: float = 0.25
    contradiction_topic_threshold: float = 0.8
    contradiction_weight_penalty: float = 0.25
    decay_rate: float = 0.05
    # Fallback text for the "user model" meta-node. This is PERSONA
    # content, not a memory parameter — it only exists because
    # get_user_model_content needs something to return before the
    # application has written its own. Anything meaningful here comes from
    # the application; the default is deliberately neutral.
    default_user_model: str = "The person I am talking to. I know nothing about them yet."
    edge_activation_decay: float = 0.6
    edge_activation_threshold: float = 0.3
    edge_boost_step: float = 0.15
    edge_decay_rate: float = 0.08
    edge_forget_threshold: float = 0.03
    edge_initial_weight: float = 0.2
    edge_max_hop_nodes: int = 3
    embeddings_enabled: bool = True
    embedding_model_path: str = '/var/www/mindnumbness/storage/models/navec_hudlit_v1_12B_500K_300d_100q.tar'
    forget_threshold: float = 0.05
    lexical_acquisition_enabled: bool = True
    lexical_age_t0: float = 2592000.0
    lexical_max_tokens_per_input: int = 20
    lexical_min_token_length: int = 2
    memory_fuzzy_weight: float = 0.1
    # Ceiling of the decay floor: the weight a node with fully earned
    # approval (reward_expectation = 1.0) settles at instead of vanishing.
    # Zero disables the floor and restores the previous behaviour.
    memory_floor_max: float = 0.25
    memory_keyword_weight: float = 0.3
    memory_min_keyword_length: int = 3
    memory_search_threshold: float = 0.3
    memory_semantic_weight: float = 0.5
    memory_weight_influence: float = 0.15
    retrospective_correction_enabled: bool = True
    retrospective_reversal_strength: float = 1.6
    retrospective_time_window_seconds: float = 180.0
    retrospective_window_size: int = 4
    reward_expectation_learning_rate: float = 0.3
    reward_min_learning_scale: float = 0.15
    reward_negative_penalty: float = 0.3
    reward_positive_boost: float = 0.2
    reward_positive_freshness_bonus: float = 600.0
    reward_preference_weight: float = 0.35
    sleep_abstract_node_weight: float = 0.75
    sleep_archive_weight_multiplier: float = 0.3
    sleep_hub_min_edge_weight: float = 0.5
    sleep_max_cluster_spokes: int = 2
    sleep_min_cluster_spokes: int = 2
    sleep_orphan_weight_threshold: float = 0.3
    stability_growth_factor: float = 1.5
    stability_initial: float = 1.0
    plasticity_stress_modifier: float = 0.25
    # Candidate pool for the fuzzy comparison. Fuzzy similarity is
    # expensive (measured: 82% of search time), so it runs on the best
    # candidates from the cheap pre-filter rather than on every node.
    # Setting the minimum above the node count restores the exhaustive
    # scan — which is exactly how the test verifies this trade-off.
    search_candidate_multiplier: int = 20
    search_candidate_minimum: int = 50
    stability_max: float = 40.0
    stm_capacity: int = 16
    stm_emotional_threshold: float = 0.6
    stm_structural_threshold: float = 0.55
    stm_structural_weight: float = 0.5
    # How many content words an utterance needs to surprise at full
    # strength. Below that, surprise is scaled down proportionally.
    surprise_full_content_tokens: int = 3
    surprise_lexical_weight: float = 0.6
    surprise_structural_weight: float = 0.4
    syllable_node_initial_weight: float = 0.1
    syllable_node_reinforce_step: float = 0.03
    syllable_word_edge_weight: float = 0.45
    vocabulary_mastery_min_weight: float = 0.18
    word_cooccurrence_edge_weight: float = 0.12
    word_node_initial_weight: float = 0.15
    word_node_reinforce_step: float = 0.04

    # Path to the database file. The only field here that is not a
    # behavioural parameter — but Database needs it, and dragging in a
    # global config just for that would defeat the point.
    db_path: str = "memory.db"

    # Names spelled differently in config.py. Kept deliberately short:
    # diverging names are a source of silent mismatches.
    _ALIASES = {"db_path": "BRAIN_DB_PATH"}

    @classmethod
    def from_module(cls, module: Any) -> "MemorySettings":
        """
        Builds settings from a module of UPPER_CASE constants. Missing
        fields keep their defaults — an application may override only part
        of them.
        """
        values = {}
        for f in fields(cls):
            name = cls._ALIASES.get(f.name, f.name.upper())
            if hasattr(module, name):
                values[f.name] = getattr(module, name)
        return cls(**values)
